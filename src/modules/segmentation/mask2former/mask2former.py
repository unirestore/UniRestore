import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .base import Base
from .prior_interface import PriorInterface, PriorInterface_v2


# /mnt/VENV/user-conda/mrchang87/miniconda3/envs/diff/lib/python3.8/site-packages/transformers/models/mask2former/
class Mask2Former(Base):
    def __init__(self, args, cfg):
        self.cfg = cfg
        super().__init__()

    def get_model(self, load_pretrain=True):
        hf_path = self.cfg["Task Model"]
        self.processor = AutoImageProcessor.from_pretrained(hf_path)
        self.processor.do_resize = False
        self.processor.do_rescale = False
        self.processor.num_labels = self.processor.num_labels + 1
        model = Mask2FormerForUniversalSegmentation.from_pretrained(hf_path)

        model = PriorInterface_v2(
            self.cfg["CPEN mode"],
            model,
            prior_layers=self.cfg["Prior Layers"],
        )

        return model

    def save_pretrained(self, output_dir, from_pt=True):
        self.model.model.save_pretrained(output_dir, from_pt)

    def encode(self, x, prior=None):
        features = self.model.encode(x, prior=prior)
        return features

    def decode(self, feature, mask_labels=None, class_labels=None):
        return self.model.model.decode(
            feature, mask_labels=mask_labels, class_labels=class_labels
        )

    def teacher_encode(self, x):  # , return_mode):
        feature = self.teacher.encode(x)
        return feature

    def teacher_decode(self, feature):  # do not compute loss
        return self.teacher.decode(feature)

    def get_mask2former_format_label(self, targets):
        targets[targets == 255] = 19
        dummy_images = [
            np.zeros((3, targets.size(1), targets.size(2)))
            for _ in range(targets.size(0))
        ]
        batch_feature = self.processor(
            images=dummy_images, segmentation_maps=targets, return_tensors="pt"
        )
        mask_labels = [label.cuda() for label in batch_feature.mask_labels]
        class_labels = [label.cuda() for label in batch_feature.class_labels]
        return mask_labels, class_labels

    def get_logits(self, outputs, shape):  # copy from post process
        class_queries_logits = (
            outputs.class_queries_logits
        )  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = (
            outputs.masks_queries_logits
        )  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=shape, mode="bilinear", align_corners=False
        )
        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = (
            masks_queries_logits.sigmoid()
        )  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        return segmentation

    def compute_pred_loss(self, pred_losses, outputs, teacher_outputs, targets):
        loss = 0
        logits = self.get_logits(outputs, targets.shape[-2:])
        teacher_logits = self.get_logits(teacher_outputs, targets.shape[-2:])

        if "mse" in pred_losses:
            criterion = torch.nn.MSELoss()
            loss += pred_losses["mse"] * criterion(logits, teacher_logits.detach())

        if "ce":
            criterion = torch.nn.CrossEntropyLoss()
            ce_loss = criterion(logits, teacher_logits.softmax(dim=1))
            loss += ce_loss
        return loss

    def compute_loss(self, losses, pred_features, teacher_features, targets):
        feat_dist_loss = self.compute_feature_dist_loss(
            losses["feature loss"], pred_features, teacher_features
        )

        with torch.no_grad():
            teacher_outputs = self.teacher_decode(teacher_features)

        if "mask2former" in losses["pred loss"]:
            mask_labels, class_labels = self.get_mask2former_format_label(targets)

            outputs = self.decode(
                pred_features, mask_labels=mask_labels, class_labels=class_labels
            )
            mask2former_loss = losses["pred loss"]["mask2former"] * outputs.loss
        else:
            outputs = self.decode(
                pred_features,
            )
            mask2former_loss = 0

        pred_loss = self.compute_pred_loss(
            losses["pred loss"], outputs, teacher_outputs, targets
        )
        loss = feat_dist_loss + mask2former_loss + pred_loss

        return loss
