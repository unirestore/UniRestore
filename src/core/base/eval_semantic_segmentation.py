from typing import Optional

import os
import torch
import torch.bin
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange, repeat
from modules.segmentation import build_segmentation_model
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex
import torchvision

from .base import BaseEvaluator
from .task import TaskMetric


class SemanticSegmentationEvaluator(BaseEvaluator):
    """
    Task-specific Trainer: SemanticSegmentation
        1. setup task tools
        2. define evaluation_step
    """

    def __init__(self, save_image: bool = False, eval_mode: str = None, need_crop: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_image = save_image
        self.eval_mode = eval_mode
        self.need_crop = need_crop

    def configure_model(self):
        super().configure_model()
        # Define task_metric
        eval_types = ["lq"]
        if self.eval_mode == "all":
            model_types = [
                "dlv3pr50",
                "dlv3pr50_ft",
                "rflwr101",
                "rflwr101_ft",
                "rflwr101_fifo",
            ]
        elif self.eval_mode == "single":
            model_types = ["dlv3pr50", "rflwr101"]
        elif self.eval_mode == "bare":
            model_types = None
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        self.task_metric = SemanticSegmentationMetric(
            eval_types=eval_types, model_types=model_types, save_seg=self.save_image
        )
        self.task_metric.requires_grad_(False).eval()

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # 1. parse batch
        lq, hq, gt, fname, task = batch
        if self.need_crop:
            lq = self.crop_tensor(lq)
            hq = self.crop_tensor(hq)
            gt = self.crop_tensor(gt)

        # 2. Inference
        preds = self.forward([lq], 'seg')
        preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds] # not used

        # 3. Visualize
        # if self.eval_mode != "bare" and batch_idx == 0:
        #     self.visualize(lq, preds)
        
        # 4. Update metrics
        self.task_metric.update_metrics(torch.cat(preds, dim=0), gt)

        # save image
        if self.save_image and len(fname) == 1:
            enh_lq = preds[0]
            logdir = self.logger.log_dir
            torchvision.utils.save_image(
                enh_lq, os.path.join(logdir, "lq", f"{fname[0]}.png")
            )
            
            seg_img = self.task_metric.seg_img[-1]
            self.task_metric.seg_img = []
            torchvision.utils.save_image(
                seg_img, os.path.join(self.logger.log_dir, "seg", f"{fname[0]}.png"))

    def on_validation_epoch_start(self) -> None:
        logdir = self.logger.log_dir
        if self.save_image:
            os.makedirs(os.path.join(logdir, "seg"), exist_ok=True)
            os.makedirs(os.path.join(logdir, "lq"), exist_ok=True)

    def on_validation_epoch_end(self):
        metrics = super().on_validation_epoch_end()
        # val_monitor
        if self.eval_mode == "bare":
            miou = metrics["val_lq/miou"]
        else:
            miou = metrics["val_lq/rflwr101"]
        # sync_dist to avoid warning
        self.log("val_monitor", miou, sync_dist=True)

    def crop_tensor(self, image, base=32):
        # center crop
        if image.ndim == 3:
            b, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            if w > 1664:
                crop_w = 1664
            return image[:, h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]
        elif image.ndim == 4:
            b, c, h, w = image.shape
            crop_h = h
            crop_w = w
            if h > 960:
                crop_h = 960
            if w > 1664:
                crop_w = 1664
            return image[:, :, h//2 - crop_h//2 : h//2 + crop_h//2, w//2 - crop_w//2 : w//2 + crop_w//2]
        else:
            raise NotImplemented

    def visualize(self, img, preds, n_img=8):
        if self.trainer.is_global_zero and self.logger:
            writer = self.logger.experiment
            # # log input only once
            if self.global_step == 0:
                vis = img[:n_img]
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


class SemanticSegmentationLoss(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        # downstream task
        self.model = build_segmentation_model(model_type)
        self.model.requires_grad_(False).eval()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # input should be [0, 1] float tensor
        logits = self.model(input)
        logits = F.interpolate(
            logits, size=input.shape[-2:], mode="bilinear", align_corners=True
        )
        loss = F.cross_entropy(logits, target, ignore_index=255)
        return loss

class SemanticSegmentationMetric(TaskMetric):
    def __init__(
        self,
        eval_types: list[str] = ["hq", "lq"],
        model_types: Optional[list[str]] = None,
        save_seg: bool= False
    ):
        super().__init__(eval_types=eval_types)
        self.model_types = model_types
        if self.model_types is not None:
            self._setup_model()
        self._setup_metrics()
        self.save_seg = save_seg
        if save_seg: 
            self.seg_img = []

    def _setup_model(self):
        self.models = nn.ModuleDict()
        for model_type in self.model_types:
            self.models[model_type] = build_segmentation_model(model_type)

    def _setup_metrics(self):
        wrapper = self.metric_wrapper
        if self.model_types is None:
            metrics = {
                "miou": wrapper(MulticlassJaccardIndex(19, ignore_index=255)),
            }
        else:
            metrics = {
                name: wrapper(MulticlassJaccardIndex(19, ignore_index=255))
                for name in self.model_types
            }
        self.metrics = nn.ModuleDict(metrics)

    def update_metrics(self, preds: Tensor, target: Tensor) -> None:
        """
        Args:
            preds: (k*B, C, H, W)
            target: (B, H, W)
        """
        h, w = target.shape[-2:]
        num_outputs = len(self.eval_types)
        target = repeat(target, "b h w -> b h w k", k=num_outputs)

        # post-process
        if self.model_types is None:  # bare
            segmaps = torch.argmax(preds, dim=1)
            segmaps = rearrange(segmaps, "(k b) h w -> b h w k", k=num_outputs)
            self.metrics["miou"].update(preds, target)
        else:
            # test time augmentation
            for name, model in self.models.items():
                to_seg = []
                for scale in [1, 0.8, 0.6]:
                    seg_in = F.interpolate(
                        preds, scale_factor=scale, mode="bilinear", align_corners=True
                    )
                    seg_out = model(seg_in)
                    to_seg.append(
                        F.interpolate(
                            seg_out,
                            size=(h, w),
                            mode="bilinear",
                            align_corners=True,
                        )
                    )
                to_seg = torch.stack(to_seg).mean(0)
                to_seg = torch.argmax(to_seg, dim=1) # [B, H, W]

                if self.save_seg and ('dlv3pr50' in name): 
                    from data.dataset_seg import CityscapesPairDataset
                    dtype, device = preds.dtype, preds.device
                    train_id_to_color = CityscapesPairDataset.train_id_to_color
                    color_map = torch.zeros((1, 3, h, w), dtype=dtype, device=device)
                    to_seg[to_seg == 255] = 19
                    color_map = train_id_to_color[to_seg.cpu()]
                    color_map = torch.tensor(color_map)
                    color_map = color_map.permute(0, 3, 1, 2) / 255.0
                    self.seg_img.append(color_map)

                segmaps = rearrange(to_seg, "(k b) h w -> b h w k", k=num_outputs)
                self.metrics[name].update(segmaps, target)

class mIoU(Metric):
    """Torchmetrics mean Intersection over Union implementation.
    https://github.com/sohyun-l/fifo/blob/main/compute_iou.py
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state(
            "hist",
            default=torch.zeros((num_classes, num_classes)),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the state with new predictions and targets.
        Args:
            preds: (B, H, W)
            targets: (B, H, W)
        """
        n = self.num_classes
        preds = preds.flatten()
        targets = targets.flatten()

        k = (targets >= 0) & (targets < n)  # filter out background
        self.hist += torch.bincount(
            n * targets[k].to(int) + preds[k], minlength=n**2
        ).reshape(n, n)

    def compute(self) -> Tensor:
        """Aggregate state over all processes and compute the metric."""

        hist = self.hist
        diag = torch.diag(hist)
        per_class_iou = diag / (hist.sum(dim=1) + hist.sum(dim=0) - diag)
        return torch.nanmean(per_class_iou)
