# Adapted from https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/datasets/open_images.py

import numpy as np
import cv2
import os
import pickle
import gzip
import pandas as pd
import torch
from tqdm import tqdm
from scipy import ndimage

class GlobalSettings:
    batch_size = None
    num_classes = None


class OpenImagesDataset:

    def __init__(self, root, cache,
                 transform=None, target_transform=None,
                 dataset_type='train', balance_data=False):
        self.root = root
        self.cache = cache
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        if self.dataset_type == 'test':
            self.sub_dir = 'test'
        else:
            self.sub_dir = 'train'

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()

        self.class_stat = None
        self.human_face = 'Human face'

    def __getitem__(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = image_info['boxes']
        if boxes is not None:
            boxes[:, 0] *= image.shape[1]
            boxes[:, 1] *= image.shape[0]
            boxes[:, 2] *= image.shape[1]
            boxes[:, 3] *= image.shape[0]
        labels = image_info['labels']
        names = image_info['names']
        if False:
        #if self.human_face in names:
            if self.dataset_type == 'train':
                # Introduce random face blur.
                if np.random.randint(2) == 0:
                    sigma_h = np.random.randint(5, 20)
                    sigma_w = np.random.randint(5, 20)
                    self._blur_face(image, boxes, names, (sigma_h, sigma_w, 0))
            else:
                # Face blur with a fixed level.
                self._blur_face(image, boxes, names, (10, 10, 0))

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def _blur_face(self, image, boxes, names, sigma):
        for i, name in enumerate(names):
            if name != self.human_face:
                continue
            box = np.round(boxes[i]).astype(np.int32)
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            crop = ndimage.gaussian_filter(crop, sigma)
            image[y1:y2, x1:x2] = crop

    def _read_tuning_data(self, class_names, class_dict):
        tune_file = os.path.join(self.root, 'meta', 'tuning_labels.csv')
        tune_data = pd.read_csv(tune_file, header=None, names=['ImageID', 'Labels'])
        data = []
        for _, row in tune_data.iterrows():
            image_id = row['ImageID']
            label_names = row['Labels'].split(' ')
            labels = []
            for name in label_names:
                if name in class_dict:
                    labels.append(class_dict[name])
                else:
                    labels.append(0)
            data.append({
                'image_id': image_id,
                'boxes': None,
                'labels': labels,
                'names': None
            })
        return data

    def _read_test_data(self, class_names, class_dict):
        subm_file = os.path.join(self.root, 'meta', 'stage_1_sample_submission.csv')
        subm_data = pd.read_csv(subm_file)
        data = []
        for _, row in subm_data.iterrows():
            image_id = row['image_id']
            labels = []
            data.append({
                'image_id': image_id,
                'boxes': None,
                'labels': labels,
                'names': None
            })
        return data

    def _read_data(self):
        metadata_path = os.path.join(self.root, self.cache)
        metadata_file = os.path.join(metadata_path, 'metadata.pkl.gz')
        if os.path.exists(metadata_file):
             data, class_names, class_dict = pickle.load(gzip.open(metadata_file, 'rb'))
        else:
            print('Preparing metadata...')
            if not os.path.exists(metadata_path):
                os.mkdir(metadata_path)
            class_description_file = os.path.join(self.root, "meta", "class-descriptions.csv")
            class_descriptions = pd.read_csv(class_description_file, index_col=None)
            name_map = dict(class_descriptions.values)
            annotation_file = os.path.join(self.root, "meta", "train-annotations-bbox.csv")
            annotations = pd.read_csv(annotation_file)
            class_names = sorted(list(annotations['LabelName'].unique()))
            class_dict = {class_name: i for i, class_name in enumerate(class_names)}
            data = []
            for image_id, group in tqdm(annotations.groupby("ImageID")):
                boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
                labels = np.array([class_dict[name] for name in group["LabelName"]])
                names = [name_map[name] if name in name_map else '' for name in group["LabelName"]]
                data.append({
                    'image_id': image_id,
                    'boxes': boxes,
                    'labels': labels,
                    'names': names
                })
            pickle.dump((data, class_names, class_dict), gzip.open(metadata_file, 'wb'))

        if self.dataset_type == 'test':
            data = self._read_test_data(class_names, class_dict)
            return data, class_names, class_dict

        # Split into training and validation subsets.
        np.random.seed(0)
        np.random.shuffle(data)
        split = int(len(data) * 0.9)
        if self.dataset_type  == 'train':
            data = data[:split]
        elif self.dataset_type  == 'val':
            data = data[split:]
        else:
            raise NotImplementedError()
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   "Number of Images: {%d}" % len(self.data),
                   "Minimum Number of Images for a Class: {%d}" % self.min_image_num,
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append("\t{%s}: {%d}" % (class_name, num))
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = os.path.join(self.root, self.sub_dir, image_id + '.jpg')
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data


def collate_fn(batch):
    image_list = []
    target = np.zeros((len(batch), GlobalSettings.num_classes), dtype=np.float32)
    for i, (image, boxes, labels) in enumerate(batch):
        image_list.append(image)
        target[i, labels] = 1
    return torch.stack(image_list), torch.from_numpy(target)
