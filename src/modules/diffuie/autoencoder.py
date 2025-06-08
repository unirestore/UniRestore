from typing import Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from einops.layers.torch import Rearrange
from torch import Tensor

from .cfrm import AdaNAFV2, NAFBlock

def my_vae_encoder_fwd(self, sample):
    """https://github.com/GaParmar/img2img-turbo/blob/main/src/model.py"""
    sample = self.conv_in(sample)
    # down
    ####################
    self.res_samples = []
    # self.igas = []
    for idx, down_block in enumerate(self.down_blocks[:-1]):
        sample = down_block(sample)
        ####################
        if self.enable_fr:
            sample = self.fr_blocks[idx](sample)
            # sample, iga = self.fr_blocks[idx](sample)
        self.res_samples += [sample]
        # self.igas += [iga]
        ####################
    sample = self.down_blocks[-1](sample.detach())
    ####################
    # middle
    sample = self.mid_block(sample)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    """https://github.com/GaParmar/img2img-turbo/blob/main/src/model.py"""
    batch_size, _, _, _ = sample.shape
    sample = self.conv_in(sample) # (1, 512, 64, 64)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds) # (1, 512, 64, 64)
    sample = sample.to(upscale_dtype)
    # up
    ####################
    condition_input = self.task_prompts[self.task] # (T, D)
    condition_time = [condition_input.unsqueeze(0).expand(batch_size, -1, -1)] # (B, T, D)
    for idx, up_block in enumerate(self.up_blocks[:-1]):  
        '''
        modules:            task_edit, up_block | task_edit, up_block | task_edit, up_block
        latent(updated): (B, 512, 64, 64) -> (B, 512, 128, 128) -> (B, 512, 256, 256) -> (B, 256, 512, 512)
        res_sample(fix): (B, 512, 64, 64) -> (B, 256, 128, 128) -> (B, 128, 256, 256)  X
        condition:       (B, T, 512)      -> (B, T, 256)        -> (B, T, 128)
        ''' 
        # hack for enabling task-specific edit
        res_sample = self.res_samples[-idx - 1]
        sample, next_condition = self.task_editors[idx](sample, res_sample, condition_time[-1]) # prompt: (B, T, D) -> (B, T, D//2)
        condition_time.append(next_condition)
        sample = up_block(sample, latent_embeds)

    sample = self.up_blocks[-1](sample, latent_embeds) # (1, 128, 512, 512)
    ####################

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample) # (1, 3, 512, 512)
    return sample

class SkipConnectedAutoEncoder(nn.Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        fr_type: Optional[str] = None,
        tedit: Optional[dict] = None
    ):
        super().__init__()
        self.tedit_dict = tedit

        ########## vae ##########
        self.vae = vae

        ##### vae encoder #####
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__
        )
        if fr_type == "CFRM":
            self.vae.encoder.fr_blocks = nn.ModuleList(
                [
                    nn.Sequential(*[NAFBlock(128) for _ in range(1)], AdaNAFV2(128)),
                    nn.Sequential(*[NAFBlock(256) for _ in range(1)], AdaNAFV2(256)),
                    nn.Sequential(*[NAFBlock(512) for _ in range(9)], AdaNAFV2(512)),
                ]
            )
        elif fr_type is not None:
            raise ValueError("Invalid fr_type")

        ##### vae decoder #####
        if self.tedit_dict:
            self.task_list = self.tedit_dict['task']
            self.tedit_type = self.tedit_dict['type']
            self.vae.decoder.task = self.tedit_dict['task'][0]
            
            self.vae.decoder.forward = my_vae_decoder_fwd.__get__(
                self.vae.decoder, self.vae.decoder.__class__
            )
            if self.tedit_type == "TFA":
                from .taskeditor import TaskEditorV1c as TaskEditor
            else:
                raise KeyError("%s is not defined in the taskeditor!, please select ['TFA']"%(self.tedit_type))

            # Task Prompts, single token
            self.vae.decoder.task_prompts = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(self.tedit_dict['prompt_len'], 512))
                for task in self.task_list
            })

            self.vae.decoder.task_editors = nn.ModuleList([
                TaskEditor(512, 512, prompt_len=self.tedit_dict['prompt_len']), # cout, cskip
                TaskEditor(512, 256, prompt_len=self.tedit_dict['prompt_len']),
                TaskEditor(512, 128, prompt_len=self.tedit_dict['prompt_len'], last_layer=True),
            ])

        else:
            self.task_list = []
            self.tedit_type = None

    def encode(
        self, images: Tensor, enable_fr: bool = False
    ) -> tuple[Tensor, list[Tensor]]:
        """Ref: https://github.com/huggingface/diffusers/blob/v0.25.1/examples/text_to_image/train_text_to_image.py#890

        Args:
            images (torch.Tensor): shape = (b, c, h, w). value in [0, 1].
        Returns:
            latents (torch.Tensor): shape = (b, 4, h//8, w//8)
            res_samples (list of torch.Tensor): len = 3. 
        Sample: 
            images: torch.Size([1, 3, 704, 512]) 
            latents: torch.Size([1, 4, 88, 64]) 
            res_samples[0]: torch.Size([1, 128, 352, 256])
            res_samples[1]: torch.Size([1, 256, 176, 128])
            res_samples[2]: torch.Size([1, 512, 88, 64])
        """
        images = images * 2 - 1  # value in [-1, 1]

        self.vae.encoder.enable_fr = enable_fr
        latents = self.vae.encode(images, return_dict=False)[0].sample()
        res_samples = self.vae.encoder.res_samples

        latents = latents * self.vae.config.scaling_factor
        return latents, res_samples

    def decode(
        self, latents: Tensor, res_samples: list[Tensor], task: str
    ) -> Tensor:
        """
        Args:
            latents (torch.Tensor): shape = (b, c, h//8, w//8)
            encoder_res_samples (list[torch.Tensor]): list of encoder residual samples
        Returns:
            images (torch.Tensor): shape = (b, c, h, w)
        Sample:

        """
        latents = latents / self.vae.config.scaling_factor

        self.vae.decoder.res_samples = res_samples
        self.vae.decoder.task = task
        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images + 1) / 2
        return images

    def forward(self, images: Tensor, task: str):
        task = 'ir'
        # hacked vae encoder
        latents, encoder_res_samples = self.encode(images, enable_fr=True)
        # hacked vae decoder
        preds = self.decode(latents, encoder_res_samples, task)
        return preds


if __name__ == "__main__":
    import torch.nn.functional as F

    device = "cuda:0"

    model_id = "stabilityai/sd-turbo"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    print("vae", sum(p.numel() for p in vae.parameters()) / 1e6, "M")

    model = SkipConnectedAutoEncoder(
            vae, 
            fr_type="adanafv2", 
            tedit={"type":'v0', "task":['ir'], "ckpt_path": None}).to(device)

    model.requires_grad_(False).eval()
    model.vae.encoder.fr_blocks.requires_grad_(True).train()
    model.vae.decoder.task_editors.requires_grad_(True).train()

    # show n param
    print("total", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        images = torch.rand(2, 3, 512, 512, device=device)

        # encode
        l0, encoder_res_samples = model.encode(images, enable_fr=True)

        # decode
        res_samples = [sample.detach() for sample in encoder_res_samples]
        preds = model.decode(l0, res_samples, 'ir')

        # loss
        loss_fr = sum(
            [F.mse_loss(res, torch.zeros_like(res)) for res in encoder_res_samples]
        )
        loss_z = F.mse_loss(preds, torch.zeros_like(preds))
        loss = loss_z + loss_fr

    # optimize...
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(name)
