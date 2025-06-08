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
from torchvision.transforms.functional import to_pil_image

# from .utils import open_image, to_image
from .corruption import corrupt, init_corruption_function
import json
import xml.etree.ElementTree as ET

coco_cate_ditc = {1: {'supercategory': 'person', 'id': 1, 'name': 'person'}, 2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, 3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, 4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, 5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, 6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, 7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, 8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, 9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, 10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, 11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, 13: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, 14: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, 15: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, 16: {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, 17: {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, 18: {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, 19: {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, 20: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, 21: {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, 22: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, 23: {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, 24: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, 25: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, 27: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, 28: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, 31: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, 32: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, 33: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, 34: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, 35: {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, 36: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, 37: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, 38: {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, 39: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, 40: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, 41: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, 42: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, 43: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, 44: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, 46: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, 47: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, 48: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, 49: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, 50: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, 51: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, 52: {'supercategory': 'food', 'id': 52, 'name': 'banana'}, 53: {'supercategory': 'food', 'id': 53, 'name': 'apple'}, 54: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, 55: {'supercategory': 'food', 'id': 55, 'name': 'orange'}, 56: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, 57: {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, 58: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, 59: {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, 60: {'supercategory': 'food', 'id': 60, 'name': 'donut'}, 61: {'supercategory': 'food', 'id': 61, 'name': 'cake'}, 62: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, 63: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, 64: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, 65: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, 67: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, 70: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, 72: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, 73: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, 74: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, 75: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, 76: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, 77: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, 78: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, 79: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, 80: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, 81: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, 82: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, 84: {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, 85: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, 86: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, 87: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, 88: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, 89: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, 90: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}}

class DetImageData(data.Dataset):
    # [degradation, clean, label]
    def __init__(self, listfile: str) -> None:
        super().__init__()
        self.listfile = listfile
        self.paths = []
        with open(listfile) as fin:
            for idx, line in enumerate(fin):
                line = line.strip().split()
                if os.path.isfile(line[0]): # lq
                    try:
                        _ = read_image(line[0], mode=ImageReadMode.RGB)
                        self.paths.append(line)
                    except:
                        continue
                elif os.path.isfile(line[1]): # hq
                    try:
                        _ = read_image(line[1], mode=ImageReadMode.RGB)
                        self.paths.append(line)
                    except:
                        continue
        self.paths = sorted(self.paths)
        print("ImageDataset:", listfile, len(self.paths))

    def __getitem__(self, index:int):
        lq_pth, hq_pth, label = self.paths[index]
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

class CoCoCorruptDataset(data.Dataset):
    COCO_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                             'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
                             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 
                             'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                             'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 
                             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                             'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                             'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 
                             'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                             'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    CoCoclass2CoCoids = {cls: idx for idx, cls in enumerate(COCO_classes)}
    RTTS_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'bus', 'motorbike'])
    RTTSclass2CoCoids = {'person':1, 'bicycle':2, 'car':3, 'bus':6, 'motorbike':4}
    RTTSclass2color = {'person': (255,0,0), 'car': (0,255,0), 'bus': (0,0,255), 'bicycle': (255,255,0), 'motorbike': (0,255,255)}

    def __init__(self, dataset, resolution = 512, is_train = True, crp_mode="common", ann="CoCo", *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.resolution = resolution
        self.is_train = is_train
        self.ann = ann
        if self.ann == "CoCo":
            self.class_name = self.COCO_classes
            self.class_mapping = self.CoCoclass2CoCoids
        elif self.ann == "RTTS":
            self.class_name = self.RTTS_classes
            self.class_mapping = self.RTTSclass2CoCoids
        else:
            raise KeyError("Currently, support for only 'CoCo' and 'RTTS'")
        self.corruption_funcs = init_corruption_function(crp_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        _, hq_pth, ann_pth, fname = self.dataset[index] # [lq, none, label, fname]

        hq = read_image(hq_pth, mode=ImageReadMode.RGB)        
        lq = hq.clone()
        label = {'boxes': [], 'labels': []}
        with open(ann_pth, 'r') as json_file:
            data_dict = json.load(json_file)
            for k, v in data_dict.items():
                if 'object' in k:
                    if (v['bndbox']["xmax"] > v['bndbox']["xmin"]) and (v['bndbox']["ymax"] > v['bndbox']["ymin"]):
                        label['boxes'].append([v['bndbox']["xmin"], v['bndbox']["ymin"], 
                                               v['bndbox']["xmax"], v['bndbox']["ymax"]])
                        label['labels'].append(self.class_mapping[v['name']])     
        # check bbox
        if len(label['boxes']) == 0:
            raise KeyError("{} don't have the vaild annotation!")
        hq, lq = TF.to_image(hq), TF.to_image(lq)
        # Transformation
        if self.is_train:
            hq, lq, label = self.pair_aug_transform(hq, lq, label)

        # generate corruption sample
        corruption_type = np.random.choice(self.corruption_funcs)
        severity = np.random.choice(5, p=[0.05, 0.25, 0.4, 0.25, 0.05]) + 1
        lq = self._degrade_image(lq, corruption_type, severity)

        hq = TF.to_dtype(hq, torch.float32, scale=True)
        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label['boxes'] = torch.tensor(label['boxes'], dtype=torch.float32)
        label['labels'] = torch.tensor(label['labels'], dtype=torch.int64) 
        return lq, hq, label, fname, 'det'

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
        # RandomSize
        w, h = to_pil_image(hq).size
        short_edge = min(w, h)
        min_ratio = 0.8
        if short_edge*min_ratio < self.resolution:
            min_ratio = self.resolution / short_edge + 0.1
        rw, rh = random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3)), random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3))
        nw, nh = int(w*rw), int(h*rh)
        resize = T.Resize([nh,nw])
        hq = resize(hq)
        lq = resize(lq)
        resize_boxes = []
        for idx, box in enumerate(label['boxes']):
            x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
            nx0, ny0, nx1, ny1 = x0*rw, y0*rh, x1*rw, y1*rh
            resize_boxes.append([nx0, ny0, nx1, ny1])
        label['boxes'] = resize_boxes

        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            
            original_w = nw
            flipped_boxes = []
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
                
                # Flip x coordinate
                new_x0 = original_w - x1 - 1
                new_x1 = original_w - x0 - 1
                flipped_boxes.append([new_x0, y0, new_x1, y1])
            label['boxes'] = flipped_boxes

        # RandomCrop
        cropped_boxes = []
        cropped_id = []
        # label = TF.crop(label, i, j, h, w)
        while len(cropped_boxes) == 0:
            i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
            cropped_hq = TF.crop(hq, i, j, h, w)
            cropped_lq = TF.crop(lq, i, j, h, w)
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box
                # Adjust the box coordinates based on the crop region
                new_x0 = x0 - j # max(0, x0 - j) 
                new_y0 = y0 - i
                new_x1 = x1 - j
                new_y1 = y1 - i
                if (new_x0 < w and new_y0 < h) and (new_x1 > 0 and new_y1 > 0) and (new_x1 > new_x0 and new_y1 > new_y0): # bounding box in the patch
                    new_x0 = max(0, new_x0) 
                    new_y0 = max(0, new_y0) 
                    new_x1 = min(new_x1, w)
                    new_y1 = min(new_y1, h)
                    cropped_boxes.append([new_x0, new_y0, new_x1, new_y1])
                    cropped_id.append(label['labels'][idx])
        
        label['boxes'] = cropped_boxes
        label['labels'] = cropped_id
        hq = cropped_hq
        lq = cropped_lq
        return hq, lq, label

class CoCoPairDataset(data.Dataset):
    COCO_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                             'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
                             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 
                             'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                             'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 
                             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                             'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                             'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 
                             'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                             'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    CoCoclass2CoCoids = {cls: idx for idx, cls in enumerate(COCO_classes)}
    RTTS_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'bus', 'motorbike'])
    RTTSclass2CoCoids = {'person':1, 'bicycle':2, 'car':3, 'bus':6, 'motorbike':4}
    RTTSclass2color = {'person': (255,0,0), 'car': (0,255,0), 'bus': (0,0,255), 'bicycle': (255,255,0), 'motorbike': (0,255,255)}

    def __init__(self, dataset, resolution = 512, is_train = True, crp_mode="common", ann="CoCo", *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.resolution = resolution
        self.is_train = is_train
        self.ann = ann

        if self.ann == "CoCo":
            self.class_name = self.COCO_classes
            self.class_mapping = self.CoCoclass2CoCoids
        elif self.ann == "RTTS":
            self.class_name = self.RTTS_classes
            self.class_mapping = self.RTTSclass2CoCoids
        else:
            raise KeyError("Currently, support for only 'CoCo' and 'RTTS'")
        self.corruption_funcs = init_corruption_function(crp_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        lq_pth, hq_pth, ann_pth, fname = self.dataset[index] # [lq, none, label, fname]

        hq = read_image(hq_pth, mode=ImageReadMode.RGB)        
        lq = read_image(lq_pth, mode=ImageReadMode.RGB)        
        label = {'boxes': [], 'labels': []}
        with open(ann_pth, 'r') as json_file:
            data_dict = json.load(json_file)
            for k, v in data_dict.items():
                if 'object' in k:
                    if (v['bndbox']["xmax"] > v['bndbox']["xmin"]) and (v['bndbox']["ymax"] > v['bndbox']["ymin"]):
                        label['boxes'].append([v['bndbox']["xmin"], v['bndbox']["ymin"], 
                                               v['bndbox']["xmax"], v['bndbox']["ymax"]])
                        label['labels'].append(self.class_mapping[v['name']])     
        # check bbox
        if len(label['boxes']) == 0:
            raise KeyError("{} don't have the vaild annotation!")

        hq, lq = TF.to_image(hq), TF.to_image(lq)
        # Transformation
        if self.is_train:
            hq, lq, label = self.pair_aug_transform(hq, lq, label)

        hq = TF.to_dtype(hq, torch.float32, scale=True)
        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label['boxes'] = torch.tensor(label['boxes'], dtype=torch.float32)
        label['labels'] = torch.tensor(label['labels'], dtype=torch.int64) 
        return lq, hq, label, fname, 'det'

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
        # RandomHorizontalFlip
        if random.random() > 0.5:
            hq = TF.hflip(hq)
            lq = TF.hflip(lq)
            
            original_w = hq.size[0]
            flipped_boxes = []
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
                
                # Flip x coordinate
                new_x0 = original_w - x1 - 1
                new_x1 = original_w - x0 - 1
                flipped_boxes.append([new_x0, y0, new_x1, y1])
            label['boxes'] = flipped_boxes

        # RandomSize
        w, h = hq.size
        short_edge = min(w, h)
        min_ratio = 0.8
        if short_edge*min_ratio < self.resolution:
            min_ratio = self.resolution / short_edge + 0.1
        rw, rh = random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3)), random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3))
        nw, nh = int(w*rw), int(h*rh)
        resize = T.Resize([nh,nw])
        hq = resize(hq)
        lq = resize(lq)
        resize_boxes = []
        for idx, box in enumerate(label['boxes']):
            x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
            nx0, ny0, nx1, ny1 = x0*rw, y0*rh, x1*rw, y1*rh
            resize_boxes.append([nx0, ny0, nx1, ny1])
        label['boxes'] = resize_boxes

        # RandomCrop
        cropped_boxes = []
        cropped_id = []
        label = TF.crop(label, i, j, h, w)
        while len(cropped_boxes) == 0:
            i, j, h, w = T.RandomCrop.get_params(hq, output_size=(self.resolution, self.resolution))
            cropped_hq = TF.crop(hq, i, j, h, w)
            cropped_lq = TF.crop(lq, i, j, h, w)
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box
                # Adjust the box coordinates based on the crop region
                new_x0 = x0 - j # max(0, x0 - j) 
                new_y0 = y0 - i
                new_x1 = x1 - j
                new_y1 = y1 - i
                if (new_x0 < w and new_y0 < h) and (new_x1 > 0 and new_y1 > 0) and (new_x1 > new_x0 and new_y1 > new_y0): # bounding box in the patch
                    new_x0 = max(0, new_x0) 
                    new_y0 = max(0, new_y0) 
                    new_x1 = min(new_x1, w)
                    new_y1 = min(new_y1, h)
                    cropped_boxes.append([new_x0, new_y0, new_x1, new_y1])
                    cropped_id.append(label['labels'][idx])
        
        label['boxes'] = cropped_boxes
        label['labels'] = cropped_id
        hq = cropped_hq
        lq = cropped_lq
        return hq, lq, label

class CoCoRealDataset(data.Dataset):
    COCO_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                             'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
                             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 
                             'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                             'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 
                             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                             'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                             'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 
                             'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                             'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    CoCoclass2CoCoids = {cls: idx for idx, cls in enumerate(COCO_classes)}
    RTTS_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'bus', 'motorbike'])
    RTTSclass2CoCoids = {'person':1, 'bicycle':2, 'car':3, 'bus':6, 'motorbike':4}
    RTTSclass2color = {'person': (255,0,0), 'car': (0,255,0), 'bus': (0,0,255), 'bicycle': (255,255,0), 'motorbike': (0,255,255)}

    def __init__(self, dataset, resolution = 512, is_train = True, crp_mode="common", ann="CoCo", *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.resolution = resolution
        self.is_train = is_train
        self.ann = ann

        if self.ann == "CoCo":
            self.class_name = self.COCO_classes
            self.class_mapping = self.CoCoclass2CoCoids
        elif self.ann == "RTTS":
            self.class_name = self.RTTS_classes
            self.class_mapping = self.RTTSclass2CoCoids
        else:
            raise KeyError("Currently, support for only 'CoCo' and 'RTTS'")
        self.corruption_funcs = init_corruption_function(crp_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        lq_pth, _, ann_pth, fname = self.dataset[index] # [lq, none, label, fname]

        lq = read_image(lq_pth, mode=ImageReadMode.RGB)        
        label = {'boxes': [], 'labels': []}
        with open(ann_pth, 'r') as json_file:
            data_dict = json.load(json_file)
            for k, v in data_dict.items():
                if 'object' in k:
                    if (v['bndbox']["xmax"] > v['bndbox']["xmin"]) and (v['bndbox']["ymax"] > v['bndbox']["ymin"]):
                        label['boxes'].append([v['bndbox']["xmin"], v['bndbox']["ymin"], 
                                               v['bndbox']["xmax"], v['bndbox']["ymax"]])
                        label['labels'].append(self.class_mapping[v['name']])     
        # check bbox
        if len(label['boxes']) == 0:
            raise KeyError("{} don't have the vaild annotation!")

        lq = TF.to_image(lq)
        # Transformation
        if self.is_train:
            lq, label = self.pair_aug_transform(lq, label)

        lq = TF.to_dtype(lq, torch.float32, scale=True)
        label['boxes'] = torch.tensor(label['boxes'], dtype=torch.float32)
        label['labels'] = torch.tensor(label['labels'], dtype=torch.int64) 
        return lq, np.nan, label, fname, 'det'

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

    def pair_aug_transform(self, lq, label):
        # RandomHorizontalFlip
        if random.random() > 0.5:
            lq = TF.hflip(lq)
            
            original_w = lq.size[0]
            flipped_boxes = []
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
                
                # Flip x coordinate
                new_x0 = original_w - x1 - 1
                new_x1 = original_w - x0 - 1
                flipped_boxes.append([new_x0, y0, new_x1, y1])
            label['boxes'] = flipped_boxes

        # RandomSize
        w, h = hq.size
        short_edge = min(w, h)
        min_ratio = 0.8
        if short_edge*min_ratio < self.resolution:
            min_ratio = self.resolution / short_edge + 0.1
        rw, rh = random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3)), random.uniform(max(min_ratio, 0.8), max(min_ratio, 1.3))
        nw, nh = int(w*rw), int(h*rh)
        resize = T.Resize([nh, nw])
        lq = resize(lq)
        resize_boxes = []
        for idx, box in enumerate(label['boxes']):
            x0, y0, x1, y1 = box # [xmin, ymin, xmax, ymax]
            nx0, ny0, nx1, ny1 = x0*rw, y0*rh, x1*rw, y1*rh
            resize_boxes.append([nx0, ny0, nx1, ny1])
        label['boxes'] = resize_boxes

        # RandomCrop
        cropped_boxes = []
        cropped_id = []
        label = TF.crop(label, i, j, h, w)
        while len(cropped_boxes) == 0:
            i, j, h, w = T.RandomCrop.get_params(lq, output_size=(self.resolution, self.resolution))
            cropped_lq = TF.crop(lq, i, j, h, w)
            for idx, box in enumerate(label['boxes']):
                x0, y0, x1, y1 = box
                # Adjust the box coordinates based on the crop region
                new_x0 = x0 - j # max(0, x0 - j) 
                new_y0 = y0 - i
                new_x1 = x1 - j
                new_y1 = y1 - i
                if (new_x0 < w and new_y0 < h) and (new_x1 > 0 and new_y1 > 0) and (new_x1 > new_x0 and new_y1 > new_y0): # bounding box in the patch
                    new_x0 = max(0, new_x0) 
                    new_y0 = max(0, new_y0) 
                    new_x1 = min(new_x1, w)
                    new_y1 = min(new_y1, h)
                    cropped_boxes.append([new_x0, new_y0, new_x1, new_y1])
                    cropped_id.append(label['labels'][idx])
        
        label['boxes'] = cropped_boxes
        label['labels'] = cropped_id
        lq = cropped_lq
        return lq, label

def custom_collate_fn(batch):
    lq, hq, label, fname = [], [], [], []
    for item in batch:
        lq.append(item[0].unsqueeze(0))
        if isinstance(item[1], torch.Tensor):
            hq.append(item[1].unsqueeze(0))
        else:
            hq.append(np.nan)
        label.append(item[2])
        fname.append(item[3])

    # cat
    lq = torch.cat(lq, dim=0)
    if isinstance(hq[0], torch.Tensor):
        hq = torch.cat(hq, dim=0)
    return lq, hq, label, fname, 'det'

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
