# Inclusive Images Challenge

Pytorch implementation of classifier

NOTE: currently uses 599 classes given in train-annotations-bbox.csv

# Installation

```
pip install torch numpy torchvision opencv-python pandas sklearn tqdm scipy
```

# Usage

## Data

The data is expected in the following directory format:

```
<root>
   |
   |----train
   |     |----images
   |
   |----test
   |     |----images
   |
   |----meta
          |
          |----train-annotations-bbox.csv
          |
          |----class-descriptions.csv
```

## Training

e.g.
```
python main.py root/
```

The first run prepares the metadata and caches it (slow).

For a complete list of options:
```
python main.py --help
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--weight-decay W] [--print-freq N] [--resume PATH]
               [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
               [--seed SEED] [--gpu GPU] [--num-classes N] [--im-size N]
               [--balance-data]
               DIR
```
