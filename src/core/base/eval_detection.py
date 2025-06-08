from typing import Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from einops import rearrange, repeat
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
import torchvision.models as models
import torchvision
import timm
from transformers import pipeline
from .base import BaseEvaluator
from .task import TaskMetric

from einops import rearrange, repeat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch import Tensor
import numpy as np 
from PIL import Image

from torchmetrics.detection import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.ops as ops

class DetectionEvaluator(BaseEvaluator):
    """
    Task-specific Evaluator: Detection
        1. setup task tools
        2. define evaluation_step
    """

    def __init__(self, save_image: bool = False, eval_mode: str = 'single', val_type='inference', score_threshold=0.995, iou_threshold=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_image = save_image
        self.eval_mode = eval_mode
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.val_type = val_type

    def configure_model(self):
        super().configure_model()
        # Define task_metric
        eval_types = ["lq"]
        if self.eval_mode == "all":
            model_types = ['retinanet', 'fastrcnn']
        elif self.eval_mode == "single":
            model_types = ['retinanet']
        elif model_type in ['retinanet', 'fastrcnn']:
            model_types = [model_type]
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        self.task_metric = DetectionMetric(
            eval_types = eval_types, 
            model_types = model_types, 
            score_threshold = self.score_threshold,
            iou_threshold = self.iou_threshold,
            val_type = self.val_type, 
            save_det = self.save_image)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # 1. parse batch
        lq, hq, gt, fname, task = batch
        
        # 2. Inference
        preds = self.forward([lq], 'det') # default: "det"
        preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds]
        
        # 3. Visualize
        # if self.eval_mode != "bare" and batch_idx == 0:
        #     self.visualize(hq, lq, preds)

        # 4. Update metrics
        self.task_metric.update_metrics(torch.cat(preds, dim=0), gt)

        # 5. save image
        if self.save_image and len(fname) == 1:
            enh_lq = preds[0]
            logdir = self.logger.log_dir

            torchvision.utils.save_image(
                enh_lq, os.path.join(logdir, "lq", f"{fname[0]}.png"))

            det_img = self.task_metric.det_img[-1]
            self.task_metric.det_img = []
            im = to_pil_image(det_img)
            im.save(os.path.join(logdir, "det", f"{fname[0]}.png"))

    def on_validation_epoch_start(self) -> None:
        logdir = self.logger.log_dir
        if self.save_image:
            os.makedirs(os.path.join(logdir, "det"), exist_ok=True)
            os.makedirs(os.path.join(logdir, "lq"), exist_ok=True)

    def on_validation_epoch_end(self):
        metrics = super().on_validation_epoch_end()
        # val_monitor
        map = metrics["val_lq/map"]
        # sync_dist to avoid warning
        self.log("val_monitor", map, sync_dist=True)

    def crop_tensor(self, image, base=16):
        # center crop
        if image.ndim == 3:
            b, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            # else:
            #     crop_h = (h // base) * base
            if w > 1664:
                crop_w = 1664
            # else:
            #     crop_w = (w // base) * base
            return image[:, 0: crop_h, 0: crop_w]
        elif image.ndim == 4:
            b, c, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            # else:
            #     crop_h = (h // base) * base
            if w > 1664:
                crop_w = 1664
            # else:
            #     crop_w = (w // base) * base
            return image[:, :, 0: crop_h, 0: crop_w]
        else:
            raise NotImplemented
            
    def visualize(self, hq, lq, preds, n_img=8):
        if self.trainer.is_global_zero and self.logger:
            writer = self.logger.experiment
            # log input only once
            if self.global_step == 0:
                vis = torch.cat([hq[:n_img], lq[:n_img]])
                vis = TF.resize(vis, (256, 256)).clamp(0, 1)
                writer.add_images(
                    "val/input",
                    vis.float(),
                    global_step=self.global_step,
                    dataformats="NCHW",
                )

            # log predition
            vis = torch.cat([img[:n_img] for img in preds])
            vis = TF.resize(vis, (256, 256)).clamp(0, 1)
            writer.add_images(
                "val/preds",
                vis.float(),
                global_step=self.global_step,
                dataformats="NCHW",
            )

class DetectionLoss(nn.Module):
    def __init__(self, model_type: str, score_threshold: float = 0.995):
        super().__init__()
        self.score_threshold = score_threshold
        # downstream task
        if model_type == "retinanet":
            weights = models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            model = models.detection.retinanet_resnet50_fpn_v2(weights=weights, threshold=self.score_threshold)
            preprocess = weights.transforms()
        elif model_type == "fastrcnn":
            weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, threshold=self.score_threshold)
            preprocess = weights.transforms()
        else:
            raise ValueError(f"Unknown model type: {model_type}")            

        self.preprocess = preprocess
        self.model = model
        self.model.requires_grad_(False).train().zero_grad()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # limit batch size = 1
        self.model.train().zero_grad()
        # input should be [0, 1] float tensor
        target['boxes'] = target['boxes'][0]
        target['labels'] = target['labels'][0]
        loss_dict = self.model(self.preprocess(input), [target])
        total_loss = sum(loss for loss in loss_dict.values())
        return total_loss

class DetectionMetric(TaskMetric):
    # FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    # RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
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
    RTTS_classes = np.array(['__background__', 'person', 'bicycle', 'car', 'bus', 'motorbike'])
    COCO2RTTSid_map = {1:1, 2:2, 3:3, 4:5, 6:4} # ['__background__', 'person', 'bicycle', 'car', 'bus', 'motorbike']
    COCOid2RTTSid = np.zeros(len(COCO_classes), dtype=np.uint8) # for training or evaluation  
    for k, v in COCO2RTTSid_map.items():
        COCOid2RTTSid[k] = v
    COCOid2RTTSclass = RTTS_classes[list(COCOid2RTTSid)] # for visualization
    RTTSclass2color = {'person': (255,0,0), 'car': (0,255,0), 'bus': (0,0,255), 'bicycle': (255,255,0), 'motorbike': (0,255,255)}

    def __init__(
        self,
        eval_types: list[str] = ["lq"],
        model_types: Optional[list[str]] = None,
        score_threshold: float = 0.995,
        iou_threshold: float = 0.5,
        val_type: int = 'inference', 
        save_det: bool = False
    ):
        super().__init__(eval_types=eval_types)
        self.model_types = model_types
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.val_type = val_type
        if self.model_types is not None:
            self._setup_model()
        self._setup_metrics()
        self.save_det = save_det
        if save_det: 
            self.det_img = []

    def _setup_model(self):
        self.detections = nn.ModuleDict()
        for model_type in self.model_types:
            print(model_type)
            if model_type == "retinanet":
                weights = models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                model = models.detection.retinanet_resnet50_fpn_v2(weights=weights, threshold=self.score_threshold)
                model.eval()
                preprocess = weights.transforms()
                detector = nn.Sequential(preprocess, model)
            elif model_type == "fastrcnn":
                weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, threshold=self.score_threshold)
                model.eval()
                preprocess = weights.transforms()
                detector = nn.Sequential(preprocess, model)
            else:
                raise ValueError(f"Unknown model type: {model_type}")            
            self.detections[model_type] = detector.requires_grad_(False).eval().cuda()

    def _setup_metrics(self):
        wrapper = self.metric_wrapper
        if self.model_types is None:
            metrics = {
                "miou": MeanAveragePrecision(iou_thresholds=[self.iou_threshold]),
            }
        else:
            metrics = {
                name: MeanAveragePrecision(iou_thresholds=[self.iou_threshold])
                for name in self.model_types
            }
        self.metrics = nn.ModuleDict(metrics)

    def update_metrics(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: tensor: enh_img 
            target: list of dict: [{'labels', 'boxes'}, ..., {...}]
            !!! only support batch size = 1
        """
        num_outputs = len(self.eval_types)
        # target = repeat(target, "b -> b k", k=num_outputs)

        for name, det in self.detections.items():
            # detect = rearrange(det(preds), "(k b) n -> b n k", k=num_outputs)
            # [{'labels', 'scores', 'boxes'}, ..., {...}]
            detect = det(preds)
            self.metrics[name].update(detect, target)
            if self.save_det and ('retinanet' in name): 
                detect = detect[0]
                if self.val_type == 'inference':
                    labels = self.COCO_classes[detect['labels'].detach().cpu().numpy()] # change RTTS ids to RTTS classes
                elif self.val_type == 'RTTS':
                    labels = self.COCOid2RTTSclass[detect['labels'].detach().cpu().numpy()] # change RTTS ids to RTTS classes
                else: 
                    raise KeyError("Visualization Only for {'inference', 'RTTS'}, but get the val_type: {}".format(self.val_type))
               
                mask = ~(labels == '__background__') 
                bbox_pred = detect["boxes"][mask]      
                score_pred = detect["scores"][mask]
                labels_pred = np.array(labels)[mask]
                indices = ops.nms(bbox_pred, score_pred, 0.1)
                labels_pred = labels_pred[indices.cpu().numpy()]


                if self.val_type == 'inference':
                    colors = 'r'
                elif self.val_type == 'RTTS':
                    colors = np.zeros((len(labels_pred), 3))
                    for k, v in self.RTTSclass2color.items():
                        colors[labels_pred==k] = v
                    colors = colors.astype(np.uint8)
                    colors = list(colors)
                    colors = [tuple(arr) for arr in colors]

                box = draw_bounding_boxes(preds[0].to(torch.uint8), 
                                          boxes=bbox_pred[indices],
                                          labels=labels_pred, # label: list of str, 
                                        #   colors=colors,
                                          width=4, font_size=30)
                self.det_img.append(box.detach())

    def reset_metrics(self, reset_fid_real: bool = True):
        """
        Args:
            reset_fid_real: Reset real features for FID computation.
            Set True for sanity_checking round.
        """
        for name, mt in self.metrics.items():
            self.metrics[name].reset()
        super().reset_metrics()

    def compute_metrics(self, prefix: Optional[str] = None) -> dict[str, float]:
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        # compute
        metrics: dict[str, list[Tensor]] = {}
        for key, metric in self.metrics.items():
            results = metric.compute()
            print()
            for k, v in results.items():
                print(k, v)
                if 'map' in k:
                    metrics[key] = v   

        outputs = {
            f"{prefix}{eval_type}/{key}": float(f"{result.item():.4f}")
            for key, values in metrics.items()
            for eval_type, result in zip(self.eval_types, values)
        }
        print(outputs)
        return outputs