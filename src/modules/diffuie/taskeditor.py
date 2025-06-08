from typing import Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch.nn.functional as F

# V1: TFA
class TaskFeatureAdapter(nn.Module):
    def __init__(self, c_out=512, c_skip=256, prompt_len=1, last_layer=False):
        '''
        c_out: output channel, align to the latent features, fixed in 512
        c_skip: residual feature dimension, default: 512, 256, 128
        expansion: LoRA expansion ratio, c_skip -> c_skip*expansion -> c_skip
        prompt_len: length of condition prompt
        '''
        super().__init__()
        c_emb = c_skip # c_skip
        self.t_gate1 = nn.Conv2d(c_skip, c_emb, 1)
        self.t_gate2 = nn.Conv2d(c_emb, c_skip, 1)
        self.conv_out = nn.Conv2d(c_skip + c_out, c_out, 1)
        
        ############## Condition Gate ###########
        self.prompt_dim = c_emb
        self.prompt_len = prompt_len
        
        hidden_dim = c_emb*prompt_len
        # filter-gate, input: res_feat (B, C, H, W)
        self.filter_gate = nn.Sequential(
            nn.InstanceNorm2d(c_skip),                                  # (B, C, H, W)
            nn.Conv2d(c_skip, c_skip, kernel_size=3, padding=1),        # (B, C, H, W)
            nn.GELU(),                                                  # (B, C, H, W)
            nn.Conv2d(c_skip, hidden_dim, kernel_size=3, padding=1),    # (B, T*D, H, W) 
            nn.AdaptiveAvgPool2d(1),                                    # (B, T*D) 
        ) # -> reshape to (B, T, D), then F.softmax(x, dim=-1)          # (B, T, D)
        
        # info-gate, input: res_feat (B, C, H, W)
        self.info_gate = nn.Sequential(
            nn.InstanceNorm2d(c_skip),                                  # (B, C, H, W)
            nn.Conv2d(c_skip, c_skip, kernel_size=3, padding=1),        # (B, C, H, W)
            nn.GELU(),                                                  # (B, C, H, W)
            nn.Conv2d(c_skip, hidden_dim, kernel_size=3, padding=1),    # (B, T*D, H, W) 
            nn.AdaptiveAvgPool2d(1),                                    # (B, T*D) 
        ) # -> reshape to (B, T, D), then F.softmax(x, dim=-1)          # (B, T, D)
        
        # content_trans, input: res_feat (B, C, H, W)
        self.content_trans = nn.Sequential(
            nn.InstanceNorm2d(c_skip),                                  # (B, C, H, W)
            nn.Conv2d(c_skip, c_skip, kernel_size=3, padding=1),        # (B, C, H, W)
            nn.GELU(),                                                  # (B, C, H, W)
            nn.Conv2d(c_skip, hidden_dim, kernel_size=3, padding=1),    # (B, T*D, H, W) 
            nn.AdaptiveAvgPool2d(1),                                    # (B, T*D) 
            nn.Tanh()                                                   # (B, T*D) 
        ) # -> reshape to (B, T, D)                                     # (B, T, D) 

        # out-gate, input: condition (B, T, D), reshape to (B, T*D)
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_dim, c_emb), # (B, D)
            nn.Tanh()                     # (B, D)
        )

        self.last_layer = last_layer
        if not self.last_layer:
            self.prompt_trans = nn.Sequential(
                nn.Linear(c_emb, c_emb//2),
                nn.GELU()
            )
            
    def forward(self, x: Tensor, skip: Tensor, condition: Tensor):
        '''
        x: (B, 512, h, w)
        skip: (B, C, h, w)
        condition: (B, T, D)
        '''
        B, C, _, _ = skip.shape
        # print("0 (x, skip, condition):", x.shape, skip.shape, condition.shape)
        # Condition Gate
        f_value = self.filter_gate(skip)  # (B, C, H, W) to (B, T*D) 
        f_value = f_value.contiguous().view(B, self.prompt_len, self.prompt_dim) # (B, T, D)
        f_value = F.softmax(f_value, dim=-1)  # (B, T, D)

        i_value = self.info_gate(skip)    # (B, C, H, W) to (B, T*D)
        i_value = i_value.contiguous().view(B, self.prompt_len, self.prompt_dim) # (B, T, D)
        i_value = F.softmax(i_value, dim=-1)  # (B, T, D)

        c_value = self.content_trans(skip)   # (B, C, H, W) to (B, T*D)
        c_value = c_value.contiguous().view(B, self.prompt_len, self.prompt_dim) # (B, T, D)

        update_condition = (f_value * condition) + (i_value * c_value)     # (B, T, D)
        condition_weight = update_condition.contiguous().view(B, self.prompt_len*self.prompt_dim) # (B, T*D)
        o_value = self.out_gate(condition_weight).unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        # print("1 (f, i, c, o, cond):", f_value.shape, i_value.shape, c_value.shape, o_value.shape, new_condition.shape)
        # selective condition feature
        hidden_skip = self.t_gate1(skip)        # (B, D, H, W)
        hidden_skip = o_value * hidden_skip     # (B, D, H, W) 
        hidden_skip = self.t_gate2(hidden_skip) # (B, C, H, W)
        skip = skip + hidden_skip               # (B, C, H, W)

        # mixing
        x = x + self.conv_out(torch.cat([x, skip], dim=1)) # (B, C, H, W)

        # output condition
        new_condition = None
        if not self.last_layer:
            new_condition = self.prompt_trans(update_condition) # (B, T, D//2)

        return x, new_condition
