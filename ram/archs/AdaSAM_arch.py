# RAM++: Robust Representation Learning via Adaptive Mask for All-in-One Image Restoration
# Zilong Zhang, Chujie Qin, Chunle Guo, Yong Zhang, Chao Xue, Ming-Ming Cheng and Chongyi Li
# https://arxiv.org/abs/2509.12039
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import os
from ram.utils.registry import ARCH_REGISTRY

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
        
class PatchEmbed(nn.Module):
   
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
       
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size 
        self.patch_size = patch_size  
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) 
        self.num_patches = self.grid_size[0] * self.grid_size[1]  
        
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        
@ARCH_REGISTRY.register()
class AdaptiveMaskPixGenerator(nn.Module):
    """
    Adaptive Mask Generator
    generate adaptive mask based on semantics
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=768,
        mask_ratio=0.75,
        use_learnable_pos_emb=False
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  
        self.visible_patches = int(num_patches * (1 - mask_ratio))  
        if use_learnable_pos_emb:
            self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed_probs, std=0.02)
        else:
            import numpy as np
            pos_table = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.register_buffer('pos_embed_probs', pos_table)

        self.get_token_probs = nn.Sequential(
            Block(
                dim=embed_dim,  
                num_heads=8,  
                mlp_ratio=4.,  
                qkv_bias=False,  
                qk_scale=None,
                drop=0.1,  
                attn_drop=0.0,  
                drop_path=0.0,  
                norm_layer=nn.LayerNorm,  
                init_values=0.  
            ),
            nn.Linear(embed_dim, 1),  
            nn.Flatten(start_dim=1)  
        )
        
        self.softmax = nn.Softmax(dim=-1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_mask(self, x, orig_size):
        x = x + self.pos_embed_probs.type_as(x).to(x.device).clone()
        logits = self.get_token_probs(x)
        logits = torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        
        B = x.shape[0]
        H, W = orig_size
        grid_h, grid_w = self.patch_embed.grid_size
        patch_size = self.patch_embed.patch_size[0]
        
        p_x_reshaped = p_x.reshape(B, grid_h, grid_w)
        p_x_pixel = p_x_reshaped.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        
        p_x_flat = p_x_pixel.reshape(B, -1)
        
        num_masked_pixels = int(H * W * (1 - (self.visible_patches / self.patch_embed.num_patches)))
        
        pixel_mask_idx = torch.multinomial(p_x_flat, num_samples=num_masked_pixels, replacement=False)
        
        pixel_mask_flat = torch.zeros((B, H*W)).to(x.device, non_blocking=True)

        pixel_mask_flat.scatter_(dim=-1, index=pixel_mask_idx.long(), value=1.0)
        pixel_mask = pixel_mask_flat.reshape(B, H, W).to(torch.bool)
        
        return p_x, p_x_pixel, pixel_mask
        
    def forward(self, img):
        B_input, C_input, H_input, W_input = img.shape
        x = self.patch_embed(img)  
        B, N, C = x.shape
        
        p_x, p_x_pixel, pixel_mask = self.get_mask(x, (H_input, W_input))
        
        mask = pixel_mask.unsqueeze(1).float().contiguous()
        
        p_x_full = p_x_pixel.unsqueeze(1).contiguous()
    
        mask_token = torch.zeros((B, 3, H_input, W_input)).to(img.device)
        
        return mask, mask_token, p_x_full

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    import numpy as np
    
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)