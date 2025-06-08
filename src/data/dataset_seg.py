import os
from glob import glob

import numpy as np
import torch
import random

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image, read_image
from collections import namedtuple
from PIL import Image

# from .utils import open_image, to_image
from .corruption import corrupt, init_corruption_function

class SEGImageData(data.Dataset):
    # [degradation, clean, label]
    def __init__(self, listfile: str) -> None:
        super().__init__()
        self.listfile = listfile
        self.paths = []
        with open(listfile) as fin:
            for line in fin:
                line = line.strip().split()
                self.paths.append(line)
        self.paths = sorted(self.paths)
        print("ImageDataset:", listfile, len(self.paths))

    def __getitem__(self, index:int):
        meta = self.paths[index]
        if len(meta) == 5:  # FoggyCityscape
            foggy1, foggy2, foggy3, hq_pth, label = meta
            fname = os.path.basename(hq_pth)
            return foggy1, foggy2, foggy3, hq_pth, label, fname
        else:
            lq_pth, hq_pth, label = meta
            if lq_pth == 'None':
                lq_pth = None
            else: 
                fname = os.path.basename(lq_pth)
            
            if hq_pth == 'None':
                hq_pth = None
            else: 
                fname = os.path.basename(hq_pth)
            return lq_pth, hq_pth, label, fname

    def __len__(self):
        return len(self.paths)

class CityscapesCorruptDataset(data.Dataset):
    # cityscape encoding
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, dataset, resolution = 512, is_train = True, crp_mode="common", *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.resolution = resolution
        # self.transform = T.Compose(
        #     [
        #         T.RandomCrop((resolution, resolution)),
        #         T.RandomHorizontalFlip(),
        #     ]
        # )
        self.is_train = is_train
        self.img_types = ["randcorrupt", "fog1", "fog2", "fog3"]
        self.corruption_funcs = init_corruption_function(crp_mode)

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index: int):
        meta = self.dataset[index] # [foggy1, foggy2, foggy3, clean, label, fname], or [lq, hq, label, fname]
        if len(meta) > 5:
            fog1, fog2, fog3, hq, label, fname = meta
            img_type = np.random.choice(self.img_types)
        else: 
            lq, hq, label, fname = meta
            img_type = "randcorrupt"

        hq = read_image(hq, mode=ImageReadMode.RGB)
        label = read_image(label)        

        if "fog" in img_type:
            if img_type == "fog1":
                lq = fog1
            elif img_type == "fog2":
                lq = fog2
            elif img_type == "fog3":
                lq = fog3
            lq = read_image(lq, mode=ImageReadMode.RGB)
        else:
            lq = hq.clone()

        hq, lq, label = TF.to_image(hq), TF.to_image(lq), tv_tensors.Mask(self.encode_target(label)) 
        if self.is_train:
            hq, lq, label = self.pair_aug_transform(hq, lq, label)

        if img_type == "randcorrupt":
            corruption_type = np.random.choice(self.corruption_funcs)
            severity = np.random.choice(5, p=[0.05, 0.25, 0.4, 0.25, 0.05]) + 1
            lq = self._degrade_image(lq, corruption_type, severity)

        hq = TF.to_dtype(hq, torch.float32, scale=True)
        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label = TF.to_dtype(label, torch.int64).squeeze(0)
        return lq, hq, label, fname, 'seg'

    def _degrade_image(
        self,
        hq: torch.Tensor,
        corruption_mode: str, 
        severity: int
    ):
        """
        Resize to [patch_size//4, patch_size], corrupt, then resize back to original resolution.
        Input:
            hq: [0, 255] uint8 tensor (C, H, W)
        Output:
            lq: [0, 255] uint8 tensor (C, H, W)
        """
        if corruption_mode == "clean":
            return hq

        h, w = hq.shape[-2:]
        # random resize short edge to [resolution//4, resolution]
        size = int(torch.randint(self.resolution // 4, self.resolution, ()))
        lq = TF.resize(hq, (size,))
        # to np
        lq = lq.permute(1, 2, 0).numpy()  # [0, 255] uint8 np (H, W, C)
        # process lq image
        lq = corrupt(lq, corruption_name=corruption_mode, severity=severity)
        # to tensor
        lq = TF.to_image(lq)
        # resize back to original resolution
        lq = TF.resize(lq, (h, w))
        return lq

    def pair_aug_transform(self, hq, lq, label):
        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
        hq = TF.crop(hq, i, j, h, w)
        lq = TF.crop(lq, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            label = TF.hflip(label)
            
        return hq, lq, label

class CityscapesPairDataset(data.Dataset):
    # cityscape encoding
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, dataset, resolution = 512, is_train = True, crp_mode=None, *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.resolution = resolution
        # self.transform = T.Compose(
        #     [
        #         T.RandomCrop((resolution, resolution)),
        #         T.RandomHorizontalFlip(),
        #     ]
        # )

        self.is_train = is_train
        self.img_types = ["fog1", "fog2", "fog3"]
        self.crp_mode = crp_mode # "fog1", "fog2", "fog3" for FoggyCityscapes

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        meta = self.dataset[index] # [foggy1, foggy2, foggy3, clean, label, fname], or [lq, hq, label, fname]
        if len(meta) > 5:
            fog1, fog2, fog3, hq, label, fname = meta
            if self.crp_mode in self.img_types:
                lq = {
                    "fog1":fog1, 
                    "fog2":fog2, 
                    "fog3":fog3
                }[self.crp_mode]
            else:
                lq = np.random.choice([fog1, fog2, fog3])
        else: 
            lq, hq, label, fname = meta

        lq = read_image(lq, mode=ImageReadMode.RGB)
        hq = read_image(hq, mode=ImageReadMode.RGB)
        label = read_image(label) 
        hq, lq, label = TF.to_image(hq), TF.to_image(lq), tv_tensors.Mask(self.encode_target(label)) 

        if self.is_train:
            hq, lq, label = self.pair_aug_transform(hq, lq, label)

        hq = TF.to_dtype(hq, torch.float32, scale=True)
        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label = TF.to_dtype(label, torch.int64).squeeze(0)
        return lq, hq, label, fname, 'seg'

    def pair_aug_transform(self, hq, lq, label):
        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
        hq = TF.crop(hq, i, j, h, w)
        lq = TF.crop(lq, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            label = TF.hflip(label)
            
        return hq, lq, label

class SEGRealDataset(data.Dataset):
    '''
    Single Real World Degradation Data
    '''
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, dataset, resolution=512, is_train=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.is_train = is_train
        self.resolution = resolution
        # self.transform = T.Compose(
        #     [
        #         T.RandomCrop((resolution, resolution)),
        #         T.RandomHorizontalFlip(),
        #     ]
        # )

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index: int):
        # Read Sample
        lq, _, label, fname = self.dataset[index]
        lq = read_image(lq, mode=ImageReadMode.RGB)
        label = read_image(label)

        lq, label = TF.to_image(lq), tv_tensors.Mask(self.encode_target(label)) 
        # Training Transform
        if self.is_train:
            lq, label = self.pair_aug_transform(lq, label)
        
        # Final Transform
        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label = TF.to_dtype(label, torch.int64).squeeze(0)
        return lq, np.nan, label, fname, 'seg'

    def pair_aug_transform(self, lq, label):
        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(lq, output_size=(self.resolution, self.resolution))
        lq = TF.crop(lq, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # RandomHorizontalFlip
        if random.random() > 0.5:
            lq = TF.hflip(lq)
            label = TF.hflip(label)
            
        return lq, label

if __name__ == "__main__":
    from torchvision.utils import save_image
    data_dict = {"Cityscapes": {"train": r'/mnt/200/b/Dataset/Segmentation/Cityscapes/train.list',   # (2975, dyn_crp)
                                "val": r'/mnt/200/b/Dataset/Segmentation/Cityscapes/val.list'},      # (500, pair)
                 "FoggyCityscapes": {"train": r'/mnt/200/b/Dataset/Segmentation/FoggyCityscapes/train.list', # (2975, foggy & dyn_crp)
                                     "val": r'/mnt/200/b/Dataset/Segmentation/FoggyCityscapes/val.list'},    # (500, foggy & dyn_crp)
                 "Foggy_Zurich": {"val": r'/mnt/200/b/Dataset/Segmentation/Foggy_Zurich/val.list'},  # (40, real-world)
                 "ACDC": {"train": r'/mnt/200/b/Dataset/Segmentation/ACDC/train.list',               # (1600, real-world)
                          "val": r'/mnt/200/b/Dataset/Segmentation/ACDC/val.list'}}                  # (406, real-world)
    
    trainset = CityscapesCorruptDataset(
                SEGImageData(data_dict['Cityscapes']['train']), 
                resolution = 512, is_train = True, crp_mode = "common")
    train_dl = data.DataLoader(
            trainset,
            batch_size= 1,
            num_workers= 1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            # prefetch_factor=2,
        )

    valset = CityscapesPairDataset(
                SEGImageData(data_dict['Cityscapes']['val']),
                resolution = 512, is_train = False)
    # valset = CityscapesPairDataset(
    #             SEGImageData(data_dict['FoggyCityscapes']['val']),
    #             resolution=512, is_train=False, crp_mode='fog3')
    # valset = SEGRealDataset(
    #             SEGImageData(data_dict['Foggy_Zurich']['val']),
    #             resolution = 512, is_train = False)
    # valset = SEGRealDataset(
    #             SEGImageData(data_dict['ACDC']['val']),
    #             resolution = 512, is_train = False)
    val_dl = data.DataLoader(
        valset,
        batch_size=1,
        num_workers=1,
        # prefetch_factor=2,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    with torch.no_grad():
        print("Train", len(train_dl))
        for ct, train in enumerate(train_dl, 1):
            print("%d / %d"%(ct, len(train_dl)), end='\r')
            if ct > 10: 
                break
        print()

        print("Val", len(val_dl))        
        vis_lq, vis_hq, vis_label = [], [], []
        for ct, val in enumerate(val_dl, 1):
            print("%d / %d"%(ct, len(val_dl)), end='\r')
            lq, hq, label, fname = val 
            print(lq.shape, hq.shape, label.shape)
            vis_lq.append(lq)
            vis_hq.append(hq)
            vis_label.append(label.float())
            if ct == 8:
                break
        print()
        save_image(torch.cat(vis_lq+vis_hq+vis_label), "vis_seg.png", nrow=8)
