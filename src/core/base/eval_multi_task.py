import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from einops import rearrange, repeat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.image import FrechetInceptionDistance as FID
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.models import resnet50, swin_v2_b, vgg16, vit_b_16
from modules.segmentation import build_segmentation_model

from .base import BaseEvaluator
from .task import TaskMetric
from .eval_image_restoration import SKPSNR, SKSSIM

class MultiTaskEvaluator(BaseEvaluator):
    """
    Task-specific Trainer: ImageRestoration
        1. setup task tools
        2. define evaluation_step
    """

    def __init__(self, save_image: bool = False, eval_mode: str = 'single', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_image = save_image
        self.eval_mode = eval_mode
        self.task_list = ['ir', 'cls', 'seg']

    def configure_model(self):
        super().configure_model()
        # Define task_metric
        eval_types = ["lq"]
        self.task_metric = MultiTaskMetric(eval_types=eval_types, task_list=self.task_list)
        self.task_metric.requires_grad_(False).eval()

    def on_validation_epoch_start(self) -> None:
        logdir = self.logger.log_dir
        if self.save_image:
            os.makedirs(os.path.join(logdir, "lq"), exist_ok=True)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        # 1. parse batch
        lq, hq, gt, fname, task = batch
        
        # 2. Inference
        preds = self.forward([lq], task[0])

        # 3. Update metrics
        if task[0] == 'ir':
            preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds]
            self.task_metric.update_metrics(torch.cat(preds, dim=0), hq, 'ir')
        if task[0] == 'cls':
            preds = [pred.mul(255).round_().clamp_(0, 255).div_(255) for pred in preds]
            self.task_metric.update_metrics(torch.cat(preds, dim=0), gt, 'cls')
        elif task[0] == 'seg':
            self.task_metric.update_metrics(torch.cat(preds, dim=0), gt, 'seg')

        # 4. Visualize
        if batch_idx == 0:
            self.visualize(hq, lq, preds)

        if self.save_image and len(fname) == 1:
            enh_lq = preds[0]
            logdir = self.logger.log_dir
            torchvision.utils.save_image(
                enh_lq, os.path.join(logdir, "lq", f"{fname[0]}.png")
            )

    def on_validation_epoch_end(self):
        metrics = super().on_validation_epoch_end()

        # val_monitor
        if 'ir' in self.task_list:
            psnr = metrics["val_lq/psnr"]
            self.log("val_ir", psnr, sync_dist=True)
            print("Task['ir']: PSNR=", psnr)
            self.log("val_monitor", psnr, sync_dist=True)
        if 'cls' in self.task_list:
            acc = metrics["val_lq/acc"]
            self.log("val_cls", acc, sync_dist=True)
            print("Task['cls']: ACC=", acc)
        if 'seg'in self.task_list:
            miou = metrics["val_lq/miou"]
            self.log("val_seg", miou, sync_dist=True)
            print("Task['seg']: mIoU=", miou)
    
    def crop_tensor(self, image, base=16):
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

class MultiTaskMetric(TaskMetric):
    def __init__(self, eval_types: list[str] = ["lq"], task_list: list[str] = ["ir"]):
        super().__init__(eval_types=eval_types)
        self.preprocess = {}
        self.task_list = task_list
        # For IR
        if 'ir' in task_list:
            self.preprocess['ir'] = T.Lambda(lambda x: torch.clamp(x, 0, 1))
        # For CLS
        if 'cls' in task_list:
            self.preprocess['cls'] = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ]
            )
            self.classifiers = resnet50(weights="IMAGENET1K_V1")
        # For SEG
        if 'seg' in task_list:
            self.segmentation = build_segmentation_model('dlv3pr50')
        
        self._setup_metrics()

    def _setup_metrics(self):
        wrapper = self.metric_wrapper
        self._update_fid_real = True
        metrics = {}
        if 'ir' in self.task_list:
            metrics['ir'] = MetricCollection(
                    {
                        "psnr": wrapper(SKPSNR(data_range=1.0)),
                        "ssim": wrapper(SKSSIM(data_range=1.0)),
                        "lpips": wrapper(LPIPS(net_type="alex", normalize=True)),
                    })
            metrics['fid'] = wrapper(FID(reset_real_features=self._update_fid_real, normalize=True))
 
        if 'cls' in self.task_list:
            metrics['acc'] = wrapper(MulticlassAccuracy(num_classes=1000, top_k=1))
        
        if 'seg' in self.task_list:
            metrics['miou'] = wrapper(MulticlassJaccardIndex(19, ignore_index=255))
 
        self.metrics = nn.ModuleDict(metrics)

    def _load_ckpt(self, model: nn.Module, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ckpt, "model.")
        model.load_state_dict(ckpt)

    def update_metrics(self, preds: Tensor, target: Tensor, task: str) -> None:
        """
        Args:
            preds: (k*B, C, H, W)
            target: (B, C, H, W)
            task: 'ir', 'cls', 'seg'
        """
        num_outputs = len(self.eval_types)
        if task == 'ir':
            device = preds.device
            with torch.autocast(device_type="cuda", enabled=False):
                preds = torch.clamp(preds, 0, 1).to(torch.float32)
                target = torch.clamp(target, 0, 1).to(torch.float32)
                preds = rearrange(
                    self.preprocess['ir'](preds), "(k b) c h w -> b c h w k", k=num_outputs
                )
                target = repeat(
                    self.preprocess['ir'](target), "b c h w -> b c h w k", k=num_outputs
                )

                self.metrics["ir"].update(preds, target)
                self.metrics["fid"].update(
                    preds, real=torch.zeros(num_outputs, dtype=bool, device=device)
                )
                if self._update_fid_real:
                    self.metrics["fid"].update(
                        target, real=torch.ones(num_outputs, dtype=bool, device=device)
                    )
        
        elif task == 'cls':
            target = repeat(target, "b -> b k", k=num_outputs)
            to_clf = self.preprocess['cls'](preds)
            logit = rearrange(self.classifiers(to_clf), "(k b) n -> b n k", k=num_outputs)
            self.metrics['acc'].update(logit, target)
        
        elif task == 'seg':
            h, w = target.shape[-2:]
            target = repeat(target, "b h w -> b h w k", k=num_outputs)

            to_seg = []
            for scale in [1, 0.8, 0.6]:
                seg_in = F.interpolate(
                    preds, scale_factor=scale, mode="bilinear", align_corners=True
                )
                seg_out = self.segmentation(seg_in)
                to_seg.append(
                    F.interpolate(
                        seg_out, size=(h, w), mode="bilinear", align_corners=True,
                ))
            to_seg = torch.stack(to_seg).mean(0)
            to_seg = torch.argmax(to_seg, dim=1)
            segmaps = rearrange(to_seg, "(k b) h w -> b h w k", k=num_outputs)
            self.metrics['miou'].update(segmaps, target)
        
        else:
            raise KeyError("EvalMetric of {} is not defined!"%(task))

    def reset_metrics(self, reset_fid_real: bool = True):
        """
        Args:
            reset_fid_real: Reset real features for FID computation.
            Set True for sanity_checking round.
        """
        self._update_fid_real = reset_fid_real
        for metric in self.metrics["fid"].metrics:
            metric.reset_real_features = reset_fid_real
        super().reset_metrics()
