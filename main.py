# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Use custom source code for models
import models
from prep import TrainAugmentation,TestTransform
from data import OpenImagesDataset, collate_fn, GlobalSettings
import numpy as np
from sklearn.metrics import fbeta_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--num-classes', default=599, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--im-size', default=512, type=int, metavar='N',
                    help='image size')
parser.add_argument('--balance-data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--cache-dir', default='cache', type=str, metavar='PATH',
                    help='path to cache directory')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='run model on test set')

best_prec1 = 0


class FLoss(nn.Module):

    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, logits, labels, thresh=0.5, beta=2):
        prob = torch.sigmoid(logits)
        prob = F.relu(prob - thresh)
        prob = torch.sign(prob)

        TP = (prob * labels).sum(1)
        TN = ((1 - prob) * (1 - labels)).sum(1)
        FP = ( prob * (1 - labels)).sum(1)
        FN = ((1 - prob) * labels).sum(1)

        prec = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        F2 = (1 + beta**2) * prec * recall / (beta**2 * prec + recall + 1e-12)
        return -(F2.mean())



def main():
    global args, best_prec1
    args = parser.parse_args()
    GlobalSettings.batch_size = args.batch_size
    GlobalSettings.num_classes = args.num_classes

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
                pretrained=True, num_classes=args.num_classes, im_size=args.im_size)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
                num_classes=args.num_classes, im_size=args.im_size)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # Use imagenet values for means and stds.
    means = np.array([123, 117, 104], dtype=np.float32)
    stds = np.array([58, 57, 57], dtype=np.float32)
    train_transform = TrainAugmentation(args.im_size, means, stds)
    train_dataset = OpenImagesDataset(
                    args.data, args.cache_dir,
                    transform=train_transform, target_transform=None,
                    dataset_type="train", balance_data=args.balance_data)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    test_transform = TestTransform(args.im_size, means, stds)
    dataset_type = 'test' if args.test else 'val'
    val_dataset = OpenImagesDataset(
                    args.data, args.cache_dir,
                    transform=test_transform, target_transform=None,
                    dataset_type=dataset_type)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.test:
        test(val_loader, model)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        f2 = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        score.update(f2, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {score.val:.3f} ({score.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, score=score))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            f2 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            score.update(f2, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Score {score.val:.3f} ({score.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       score=score))

        print(' * Score {score.avg:.3f}'.format(score=score))

    return score.avg


def test(loader, model):
    model.eval()
    preds = np.zeros((len(loader.dataset), args.num_classes), dtype=np.float32)
    with torch.no_grad():
        idx = 0
        for i, (input, _) in enumerate(loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            output = model(input)
            output = torch.sigmoid(output)
            preds[idx:idx + output.shape[0]] = output.data.cpu().numpy()
            idx += output.shape[0]
    pickle.dump((preds, loader.dataset.class_names), open('preds.pkl', 'wb'))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    save_path = os.path.join(args.data, args.cache_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(args.data, args.cache_dir, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    output = torch.sigmoid(output)
    target = target.data.cpu().numpy()
    output = output.data.cpu().numpy().round()
    return fbeta_score(target , output, beta=2, average='samples')


if __name__ == '__main__':
    main()
