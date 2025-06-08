from typing import Optional

import torch
import torch.nn.functional as F
from core.base import (
    BaseTrainer,
    ClassificationEvaluator,
    ClassificationLoss,
    ImageRestorationEvaluator,
    ImageRestorationLoss,
    SemanticSegmentationEvaluator,
    SemanticSegmentationLoss,
    DetectionEvaluator,
    DetectionLoss,
    MultiTaskEvaluator
)
from modules.diffuie.unifie import DiffUIE

class LitUniFIE(BaseTrainer):
    """
    Enhancer Model Trainer:
        1. setup enhancer model
        2. define training_step
        3. define forward for inference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_kwargs = self.model_kwargs
        self.frenc: Optional[dict] = model_kwargs.get("frenc")
        self.cnet: Optional[dict] = model_kwargs.get("cnet")
        self.tedit: Optional[dict] = model_kwargs.get("tedit")
        self.task_dict = self.tedit['task']

    def configure_model(self):
        super().configure_model()
        # 1. Define model
        self.model = DiffUIE(
            frenc=self.frenc,
            cnet=self.cnet,
            tedit=self.tedit
        )

        # 2. Load ckpt & 3. Configure trainable params
        # default: freeze all
        self.model.requires_grad_(False).eval()
        self.no_checkpoint += ["model"]

        if self.frenc: 
            # CFRM
            if (ckpt_path := self.frenc["ckpt_path"]) is not None:
                ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                fr_blocks_ckpt = {
                    k[31:]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.ae.vae.encoder.fr_blocks.")
                }
                self.model.ae.vae.encoder.fr_blocks.load_state_dict(fr_blocks_ckpt)
                if self.trainer.is_global_zero:
                    print(f"!!Loaded frenc from {ckpt_path}")
            # Trainable CFRM layers
            if self.frenc["train"]:
                self.no_ckpt_exception += ["model.ae.vae.encoder.fr_blocks"]
                self.model.ae.vae.encoder.fr_blocks.requires_grad_(True).train()

        if self.cnet:
            # Controlnet & SC-Tuner
            if (ckpt_path := self.cnet["ckpt_path"]) is not None:
                ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                controller_ckpt = {
                    k[17:]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.controller.")
                }
                self.model.controller.load_state_dict(controller_ckpt)
                csc_editors_ckpt = {
                    k[29:]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.base_model.csc_editors.")
                }
                self.model.base_model.csc_editors.load_state_dict(csc_editors_ckpt)
                if self.trainer.is_global_zero:
                    print(f"!!Loaded cnet from {ckpt_path}")
            # Trainable Controlnet & SC-Tuner
            if self.cnet["train"]:
                # controller
                self.no_ckpt_exception += ["model.controller"]
                self.model.controller.requires_grad_(True).train()
                # control mechanism
                control_type = self.cnet["type"]
                if control_type == "spade":
                    self.no_ckpt_exception += ["model.base_model"]
                    for name, module in self.model.base_model.named_modules():
                        if "spade" in name:
                            module.requires_grad_(True).train()
                elif "scedit" in control_type:
                    self.no_ckpt_exception += ["model.base_model.csc_editors"]
                    self.model.base_model.csc_editors.requires_grad_(True).train()
                else:
                    raise NotImplementedError

        if self.tedit:
            # TFA
            if (ckpt_path := self.tedit["ckpt_path"]) is not None:
                ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                # task prompts
                task_prompts_weight = {
                    k[len("model.ae.vae.decoder.task_prompts."): ]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.ae.vae.decoder.task_prompts.")
                }
                self.model.ae.vae.decoder.task_prompts.load_state_dict(
                    task_prompts_weight, strict=False
                )
                print(f"!!Loaded task_prompts from {ckpt_path}")

                # task editor
                task_editor_weight = {
                    k[len("model.ae.vae.decoder.task_editors."): ]: v
                    for k, v in ckpt.items()
                    if k.startswith("model.ae.vae.decoder.task_editors.")
                }
                self.model.ae.vae.decoder.task_editors.load_state_dict(
                    task_editor_weight
                )
                print(f"!!Loaded task_editor from {ckpt_path}")

            if self.tedit['train']:
                # Trainable TFA, if introducing new task, only prompt trainable.
                self.no_ckpt_exception += ["model.ae.vae.decoder.task_editors"]
                self.no_ckpt_exception += ["model.ae.vae.decoder.task_prompts"]
                self.model.ae.vae.decoder.task_editors.requires_grad_(False).train()
                self.model.ae.vae.decoder.task_prompts.requires_grad_(True).train()
            
    def fr_training_fwd(self, hq, lq):
        # AE.Encoder + CFRM forwarding
        with torch.no_grad():
            # encode hq
            h0, h0_mids = self.model.ae.encode(hq, enable_fr=False)
            # encode lq
            if self.frenc:
                # using frenc by enabling fr
                torch.set_grad_enabled(self.frenc["train"])  # enable grad for frenc
                l0, l0_mids = self.model.ae.encode(lq, enable_fr=True)
            else:
                # using plain encoder by disabling fr
                l0, l0_mids = self.model.ae.encode(lq, enable_fr=False)
        return h0, h0_mids, l0, l0_mids

    def fr_loss_fn(self, h0, h0_mids, l0, l0_mids):
        # CFRM loss
        loss_layer1 = F.mse_loss(l0_mids[0], h0_mids[0])
        loss_layer2 = F.mse_loss(l0_mids[1], h0_mids[1])
        loss_layer3 = F.mse_loss(l0_mids[2], h0_mids[2])
        loss_enc = F.mse_loss(l0, h0)
        loss_frenc = 0.1 * loss_layer1 + 0.1 * loss_layer2 + 0.01 * loss_layer3
        # Log
        self.log_dict(
            {
                "train/loss_layer1": loss_layer1,
                "train/loss_layer2": loss_layer2,
                "train/loss_layer3": loss_layer3,
                "train/loss_enc": loss_enc,
                "train/loss_frenc": loss_frenc,
            }
        )
        return loss_frenc

    def cn_training_fwd(self, h0, l0):
        # Controlnet & SC-Tuner + denoising UNet forwarding
        with torch.no_grad():
            # diffuse dm input
            zt, _, timesteps = self.model.diffuse(h0)
            # predict z0 by cnet
            torch.set_grad_enabled(self.cnet["train"])
            pred_z0 = self.model.predict_z0(zt, conditions=l0, timesteps=timesteps)
        return pred_z0

    def cn_loss_fn(self, pred_z0, h0):
        # Control Loss
        loss_cnet = F.mse_loss(pred_z0, h0)
        self.log("train/loss_cnet", loss_cnet)
        return loss_cnet

    def te_training_fwd(self, pred_z0, l0_mids, task):
        # AE.Decoder + TFA forwarding
        bsz = pred_z0.size(0)
        if self.frenc["train"]:
            l0_mids = [feat.detach() for feat in l0_mids]
        preds = self.model.ae.decode(pred_z0.detach(), l0_mids, task)
        return preds

    def te_loss_fn(self, preds, hq, gt, task):
        # TFA Loss depend on the model class
        raise NotImplementedError

    def training_step(self, batch):
        lq, hq, gt, fname, task = batch
        task = task[0] 
        loss = loss_fr = loss_cn = loss_te = 0
        h0, h0_mids, l0, l0_mids = self.fr_training_fwd(hq, lq)

        if self.cnet:
            pred_z0 = self.cn_training_fwd(h0, l0)
        else:
            pred_z0 = l0

        if self.tedit:
            preds = self.te_training_fwd(pred_z0, l0_mids, task)

        if self.frenc and self.frenc["train"]:
            loss_fr = self.fr_loss_fn(h0, h0_mids, l0, l0_mids)
            loss += loss_fr
        if self.cnet and self.cnet["train"]:
            loss_cn = self.cn_loss_fn(pred_z0, h0)
            loss += loss_cn
        if self.tedit:
            loss_te = self.te_loss_fn(preds, hq, gt, task)
            # multi-task learning
            if len(self.task_dict) > 1 and task != 'ir':
                preds_ir = self.te_training_fwd(pred_z0, l0_mids, 'ir')
                loss_te += F.l1_loss(preds_ir, hq)
            loss += loss_te
        self.log("train/loss", loss)
        return loss

    @torch.inference_mode()
    def forward(self, inputs, task):
        """
        Args:
            inputs: [hq, lq]
        Returns:
            preds: [enh_hq, enh_lq]
        """
        outputs = [self.model.forward(imgs, task) for imgs in inputs]
        return outputs

class LitUniFIEMTL(LitUniFIE, MultiTaskEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_model(self):
        super().configure_model()
        task_list = ['ir', 'cls', 'seg']
        self.criterion = {}
        for task in task_list:
            if task == 'ir':
                self.criterion[task] = ImageRestorationLoss()
            elif task == 'cls':
                self.criterion[task] = ClassificationLoss(model_type='r50v1')
                print(f"!!Downstream CLF: r50v1")
            elif task == 'seg':
                self.criterion[task] = SemanticSegmentationLoss(model_type='dlv3pr50')
                print(f"!!Downstream SEG: dlv3pr50")
            else:
                raise KeyError("Task [{}] is not defined!"%(task))
            self.criterion[task].requires_grad_(False).eval()

    def te_loss_fn(self, preds, hq, gt=None, task=None):
        if task == 'ir':
            loss = 10*self.criterion[task](preds, hq)
        else:
            loss = 0.1*self.criterion[task](preds, gt)     
        # Log
        self.log("train/loss_{}".format(task), loss)
        return loss

class LitUniFIEIR(LitUniFIE, ImageRestorationEvaluator):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = ImageRestorationLoss()
        self.criterion.requires_grad_(False).eval()

    def te_loss_fn(self, preds, hq, gt=None, task=None):
        # Loss
        loss_ir = self.criterion(preds, hq)
        # Log
        self.log("train/loss_ir", loss_ir)
        return loss_ir

class LitUniFIECLF(LitUniFIE, ClassificationEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_model(self):
        super().configure_model()
        downstream = self.model_kwargs["downstream"]
        self.criterion = ClassificationLoss(model_type=downstream)
        self.criterion.requires_grad_(False).eval()
        if self.trainer.is_global_zero:
            print(f"!!Downstream CLF: {downstream}")

    def te_loss_fn(self, preds, hq, gt, task=None):
        # Loss
        loss_clf = self.criterion(preds, gt)
        # Log
        self.log_dict({"train/loss_clf": loss_clf})
        return loss_clf

class LitUniFIESemseg(LitUniFIE, SemanticSegmentationEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_model(self):
        super().configure_model()
        downstream = self.model_kwargs["downstream"]
        self.criterion = SemanticSegmentationLoss(model_type=downstream)
        self.criterion.requires_grad_(False).eval()
        if self.trainer.is_global_zero:
            print(f"!!Downstream SEG: {downstream}")

    def te_loss_fn(self, preds, hq, gt, task=None):
        # Loss
        loss_semseg = self.criterion(preds, gt)
        # Log
        self.log_dict({"train/loss_semseg": loss_semseg})
        return loss_semseg

class LitUniFIEDET(LitUniFIE, DetectionEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_model(self):
        super().configure_model()
        downstream = self.model_kwargs["downstream"]
        self.criterion = DetectionLoss(model_type=downstream, score_threshold = 0.995)
        self.criterion.requires_grad_(False).eval()
        if self.trainer.is_global_zero:
            print(f"!!Downstream DET: {downstream}")

    def te_loss_fn(self, preds, hq, gt, task=None):
        # Loss
        loss_det = self.criterion(preds, gt)
        # Log
        self.log_dict({"train/loss_det": loss_det})
        return loss_det