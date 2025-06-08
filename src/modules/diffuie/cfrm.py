import math

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from einops.layers.torch import Rearrange
from torch import Tensor

from .nafnet_arch import NAFBlock


class AdaNAFV2(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * 4
        groups = 16
        # in
        self.conv_in = nn.Conv2d(c, dw_channel, 1)
        self.group_norm = nn.GroupNorm(groups, dw_channel)
        self.group_conv = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=groups)
        self.gelu = nn.GELU()
        # intra group attention
        self.intra_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, dw_channel, 1, groups=groups),
        )
        # inter group attention
        self.to_gw = Rearrange("b (g k) h w -> b g k h w", g=groups)
        self.inter_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, groups, 1),
            Rearrange("b g h w -> b g 1 h w"),
        )
        self.inverse_gw = Rearrange("b g k h w -> b (g k) h w")
        # out
        self.pwconv = nn.Conv2d(dw_channel, c, 1)
        # naf block
        self.nafblock = NAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate)

    def forward(self, inp):
        x = inp

        x = self.conv_in(x)
        x = self.group_norm(x)
        x = self.group_conv(x)
        x = self.gelu(x)
        x = x * self.intra_group_attn(x)
        iga = self.inter_group_attn(x)
        x = self.inverse_gw(self.to_gw(x) * iga)
        x = self.pwconv(x)
        x = inp + x

        x = self.nafblock(x)
        return x  # , torch.mean(iga, dim=(2, 3, 4))


class AdaNAF(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        groups = 32
        # in
        self.group_norm = nn.GroupNorm(groups, c)
        self.group_conv = nn.Conv2d(c, dw_channel, 3, padding=1, groups=groups)
        self.gelu = nn.GELU()
        # intra group attention
        self.intra_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, dw_channel, 1, groups=groups),
        )
        # inter group attention
        self.to_gw = Rearrange("b (g k) h w -> b g k h w", g=groups)
        self.inter_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, groups, 1),
            Rearrange("b g h w -> b g 1 h w"),
        )
        self.inverse_gw = Rearrange("b g k h w -> b (g k) h w")
        # out
        self.pwconv = nn.Conv2d(dw_channel, c, 1)
        # naf block
        self.nafblock = NAFBlock(c, DW_Expand, FFN_Expand, drop_out_rate)

    def forward(self, inp):
        x = inp

        x = self.group_norm(x)
        x = self.group_conv(x)
        x = self.gelu(x)
        x = x * self.intra_group_attn(x)
        x = self.inverse_gw(self.to_gw(x) * self.inter_group_attn(x))
        x = self.pwconv(x)
        x = inp + x

        x = self.nafblock(x)
        return x


class AdaConv(nn.Module):
    def __init__(self, c_in, expansion=2, layer_scale_init_value=1e-6):
        super().__init__()
        c_emb = int(c_in * expansion)
        n_styles = 4
        # feature branch
        self.dwconv = nn.Conv2d(c_in, c_in, kernel_size=7, padding=3, groups=c_in)
        self.norm = nn.LayerNorm(c_in, eps=1e-6)
        self.pwconv1 = nn.Linear(c_in, c_emb)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(c_emb, c_in)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c_in)))

        # style branch
        self.fc = nn.Linear(c_in, n_styles)
        self.styles = nn.Parameter(torch.empty((n_styles, c_emb, c_in)))
        nn.init.kaiming_uniform_(self.styles, a=math.sqrt(5))

    def forward(self, x):
        input = x
        # feature branch
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # style branch
        y = torch.mean(input, dim=(2, 3))  # (N, C)
        y = torch.softmax(self.fc(y), dim=-1)  # (N, S)
        weight = torch.einsum("ns, sdc -> ndc", y, self.styles)

        x = torch.einsum("nhwd, ndc -> nhwc", x, weight)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + input


class AdaConvV2(nn.Module):
    def __init__(self, c_in, expansion=2, layer_scale_init_value=1e-6):
        super().__init__()
        self.c_emb = c_emb = int(c_in * expansion)
        self.n_styles = n_styles = 8
        # feature branch
        self.dwconv = nn.Conv2d(c_in, c_in, kernel_size=7, padding=3, groups=c_in)
        self.norm = nn.LayerNorm(c_in, eps=1e-6)
        self.pwconv1 = nn.Linear(c_in, c_emb)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(c_emb, c_in)
        self.gamma = nn.Parameter(torch.empty((c_in)))
        nn.init.constant_(self.gamma, layer_scale_init_value)

        # style branch
        self.fc = nn.Linear(c_in, n_styles)
        self.styles = nn.Parameter(torch.empty((c_emb, c_emb)))
        nn.init.kaiming_uniform_(self.styles, a=math.sqrt(5))

    def forward(self, x):
        input = x
        # feature branch
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # style branch
        y = torch.mean(input, dim=(2, 3))  # (N, C)
        y = torch.softmax(self.fc(y), dim=-1)  # (N, C)->(N, n_styles)
        # (N, n_styles)->(N, c_emb)
        y = torch.repeat_interleave(
            y, repeats=self.c_emb // self.n_styles, dim=-1, output_size=self.c_emb
        )
        weight = torch.einsum("nd, dc -> ndc", y, self.styles)

        x = torch.einsum("nhwd, ndc -> nhwc", x, weight)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + input


def my_vae_encoder_fwd(self, sample):
    """https://github.com/GaParmar/img2img-turbo/blob/main/src/model.py"""
    sample = self.conv_in(sample)
    # down
    self.res_samples = []
    for idx, down_block in enumerate(self.down_blocks):
        ####################
        if self.enable_fp:
            sample = self.fp_blocks[idx](sample)
        self.res_samples += [sample]
        ####################
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample.detach())
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class FeaturePurifyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_id = "stabilityai/sd-turbo"

        ##### vae #####
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

        ## vae encoder ##
        self.vae.encoder.forward = my_vae_encoder_fwd.__get__(
            self.vae.encoder, self.vae.encoder.__class__
        )
        self.vae.encoder.fp_blocks = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(*[NAFBlock(128) for _ in range(0)], AdaNAFV2(128)),
                nn.Sequential(*[NAFBlock(256) for _ in range(0)], AdaNAFV2(256)),
                nn.Sequential(*[NAFBlock(512) for _ in range(8)], AdaNAFV2(512)),
            ]
        )
        # self.vae.encoder.fp_blocks = nn.ModuleList(
        #     [
        #         nn.Identity(),
        #         # nn.Sequential(*[NAFBlock(128) for _ in range(1)]),
        #         nn.Sequential(*[NAFBlock(128) for _ in range(2)]),
        #         nn.Sequential(*[NAFBlock(256) for _ in range(2)]),
        #         nn.Sequential(*[NAFBlock(512) for _ in range(8)]),
        #     ]
        # )

    def encode_latents(self, images: Tensor):
        """Adapted from https://github.com/huggingface/diffusers/blob/v0.25.1/examples/text_to_image/train_text_to_image.py#890

        Args:
            images (torch.Tensor): shape = (b, c, h, w). value in [0, 1].
        Returns:
            latents (torch.Tensor): shape = (b, 4, h//8, w//8)
        """
        images = images * 2 - 1  # value in [-1, 1]
        latents = self.vae.encode(images, return_dict=False)[0].sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: Tensor):
        """
        Args:
            latents (torch.Tensor): shape = (b, c, h//8, w//8)
            encoder_res_samples (list[torch.Tensor]): list of encoder residual samples
        Returns:
            images (torch.Tensor): shape = (b, c, h, w)
        """
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images + 1) / 2
        return images

    def forward(self, images: Tensor):
        self.vae.encoder.enable_fp = True
        latents = self.encode_latents(images)
        preds = self.decode_latents(latents)
        return preds


if __name__ == "__main__":
    device = "cuda:0"

    model = FeaturePurifyEncoder()
    print(
        f"n parameters: {sum(p.numel() for p in model.vae.encoder.fp_blocks.parameters()):,}"
    )

    x = torch.rand(1, 3, 256, 256).to(device)
    model.vae.encoder.enable_fp = True
    y = model(x)
