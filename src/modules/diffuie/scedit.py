"""
Adapted from: SCEdit
model -> https://github.com/modelscope/scepter/blob/main/scepter/modules/model/tuner/sce/scetuning.py
config -> https://github.com/modelscope/scepter/tree/main/scepter/methods/scedit/ctr
"""

import torch
from torch import nn


class SCEAdapter(nn.Module):
    def __init__(self, c_in, c_emb):
        super().__init__()

        # pw-conv is slightly faster than permute+linear
        self.tuner = nn.Sequential(
            nn.Conv2d(c_in, c_emb, 1), nn.GELU(), nn.Conv2d(c_emb, c_in, 1)
        )

    def forward(self, x):
        return self.tuner(x) + x


class CSCEAdapter(nn.Module):
    def __init__(self, c_in, c_emb, c_cond):
        super().__init__()

        self.proj = nn.Conv2d(c_cond, c_in, 1)

        # pw-conv is slightly faster than permute+linear
        self.tuner = nn.Sequential(
            nn.Conv2d(c_in, c_emb, 1), nn.GELU(), nn.Conv2d(c_emb, c_in, 1)
        )

    def forward(self, x, condition):
        proj_cond = self.proj(condition)

        return self.tuner(x + proj_cond) + proj_cond + x


class CSCEAdapterV2(nn.Module):
    def __init__(self, c_in, c_emb, c_cond):
        super().__init__()
        # 1. change proj to 3x3
        # 2. add layernorm and gamma to the tuner, c.f., nafnet
        # 3. zero init last conv

        self.proj = nn.Conv2d(c_cond, c_in, 3, padding=1)

        self.tuner = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_emb),
            nn.GELU(),
            nn.Linear(c_emb, c_in),
        )
        self.gamma = nn.Parameter(torch.zeros((1, c_in, 1, 1)), requires_grad=True)

        # zero init last conv
        nn.init.zeros_(self.tuner[-1].weight)
        nn.init.zeros_(self.tuner[-1].bias)

    def forward(self, x, condition):
        inp = x
        proj_cond = self.proj(condition)

        x = (x + proj_cond).permute(0, 2, 3, 1).contiguous()
        x = self.tuner(x).permute(0, 3, 1, 2).contiguous()

        return x * self.gamma + inp + proj_cond


class CSCEAdapterV3(nn.Module):
    def __init__(self, c_in, expansion, c_cond, layer_scale_init_value=1e-6):
        super().__init__()
        # project condition to input channel size with zero conv
        # adapted from StableCascade's controlnet
        self.proj = nn.Sequential(
            nn.Conv2d(c_cond, c_cond, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_cond, c_in, kernel_size=1, bias=False),
        )
        nn.init.constant_(self.proj[-1].weight, 0)

        # convnext-like tuner
        # adated from convnext's block
        self.dwconv = nn.Conv2d(c_in, c_in, kernel_size=7, padding=3, groups=c_in)
        self.norm = nn.LayerNorm(c_in, eps=1e-6)
        self.pwconv1 = nn.Linear(c_in, int(c_in * expansion))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(c_in * expansion), c_in)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((c_in)), requires_grad=True
        )

    def forward(self, x, condition):
        input = x
        proj_cond = self.proj(condition)

        x = self.dwconv(x + proj_cond)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return x + input + proj_cond


if __name__ == "__main__":
    for MODEL in [CSCEAdapter, CSCEAdapterV2, CSCEAdapterV3]:
        print(f"MODEL: {MODEL.__name__}")
        n_params = 0
        model = MODEL(320, int(320 * 1.2), 256)
        print(f"n params: {sum(p.numel() for p in model.parameters()):,}")
        n_params += sum(p.numel() for p in model.parameters()) * 4
        model = MODEL(640, int(640 * 1.2), 256)
        print(f"n params: {sum(p.numel() for p in model.parameters()):,}")
        n_params += sum(p.numel() for p in model.parameters()) * 3
        model = MODEL(1280, int(1280 * 1.2), 256)
        print(f"n params: {sum(p.numel() for p in model.parameters()):,}")
        n_params += sum(p.numel() for p in model.parameters()) * 5
        print(f"n_params: {n_params:,}")
