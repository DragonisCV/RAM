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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
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
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
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
    """
    图像到块嵌入层
    将图像分割成不重叠的块，并投影到嵌入维度
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        初始化块嵌入层
        
        参数:
            img_size: 输入图像大小
            patch_size: 块大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        # 确保img_size和patch_size为元组
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size  # 图像大小
        self.patch_size = patch_size  # 块大小
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 网格大小
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 块数量
        
        # 卷积投影层，用卷积来实现分块和投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [B, C, H, W]
            
        返回:
            块嵌入 [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        # 检查输入图像尺寸是否正确
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}*{W}) 与模型不匹配 ({self.img_size[0]}*{self.img_size[1]})."
        
        # 使用卷积进行分块和投影
        # [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size] -> [B, H//patch_size*W//patch_size, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        
@ARCH_REGISTRY.register()
class AdaptiveMaskPixGenerator(nn.Module):
    """
    自适应掩码生成器
    生成基于输入图像内容的自适应掩码
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
        """
        初始化自适应掩码生成器
        
        参数:
            img_size: 输入图像大小
            patch_size: 块大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            mask_ratio: 掩码比率（被掩码的token比例）
            use_learnable_pos_emb: 是否使用可学习的位置嵌入
        """
        super().__init__()
        
        # 块嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # 块数量
        self.visible_patches = int(num_patches * (1 - mask_ratio))  # 可见块数量
        # print(f"自适应掩码生成器: 可见块数 = {self.visible_patches}, 总块数 = {num_patches} ({(1-mask_ratio)*100:.1f}%可见)")
        
        # 掩码预测的位置嵌入
        if use_learnable_pos_emb:
            # 可学习的位置嵌入
            self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed_probs, std=0.02)
        else:
            # 固定的正弦位置嵌入
            import numpy as np
            pos_table = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.register_buffer('pos_embed_probs', pos_table)

        # 令牌概率预测网络
        self.get_token_probs = nn.Sequential(
            Block(
                dim=embed_dim,  # 嵌入维度 
                num_heads=8,  # 注意力头数
                mlp_ratio=4.,  # MLP隐藏层维度比率
                qkv_bias=False,  # 不使用QKV偏置
                qk_scale=None,  # 使用默认缩放因子
                drop=0.1,  # Dropout率
                attn_drop=0.0,  # 注意力dropout率
                drop_path=0.0,  # 路径dropout率
                norm_layer=nn.LayerNorm,  # 归一化层
                init_values=0.  # 初始化值
            ),
            nn.Linear(embed_dim, 1),  # 投影到单一通道
            nn.Flatten(start_dim=1)  # 展平
        )
        
        # Softmax层，用于将logits转换为概率
        self.softmax = nn.Softmax(dim=-1)
        
        # 掩码令牌生成 - 创建可学习的掩码令牌
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模型权重
        
        参数:
            m: 模型模块
        """
        if isinstance(m, nn.Linear):
            # 线性层权重使用xavier均匀初始化
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 偏置初始化为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 层归一化参数初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_mask(self, x, orig_size):
        """
        基于输入特征生成自适应掩码，并扩展为像素级掩码
        返回:
            p_x: token选择概率 [B, N]
            p_x_pixel: 像素级掩码概率 [B, H, W]
            pixel_mask: 像素级布尔掩码 [B, H, W] (True = 被掩码)
        """
        # 添加位置嵌入
        x = x + self.pos_embed_probs.type_as(x).to(x.device).clone()
        # 获取token重要性得分
        logits = self.get_token_probs(x)
        logits = torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        
        # 获取原始图像尺寸和patch尺寸
        B = x.shape[0]
        H, W = orig_size
        grid_h, grid_w = self.patch_embed.grid_size
        patch_size = self.patch_embed.patch_size[0]
        
        # 将patch概率扩展为像素概率
        p_x_reshaped = p_x.reshape(B, grid_h, grid_w)
        p_x_pixel = p_x_reshaped.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        
        # 将概率展平用于像素级采样
        p_x_flat = p_x_pixel.reshape(B, -1)
        
        # 计算要mask的像素数量（保持与原来相同的掩码比例）
        num_masked_pixels = int(H * W * (1 - (self.visible_patches / self.patch_embed.num_patches)))
        
        # 对像素进行采样
        pixel_mask_idx = torch.multinomial(p_x_flat, num_samples=num_masked_pixels, replacement=False)
        
        # 创建像素掩码 (初始全为0，表示全部可见)
        pixel_mask_flat = torch.zeros((B, H*W)).to(x.device, non_blocking=True)
        # 在采样位置设置为1（表示这些位置被掩码）
        pixel_mask_flat.scatter_(dim=-1, index=pixel_mask_idx.long(), value=1.0)
        pixel_mask = pixel_mask_flat.reshape(B, H, W).to(torch.bool)
        
        return p_x, p_x_pixel, pixel_mask
        
    def forward(self, img):
        """
        自适应掩码生成器的前向传播 - 像素级掩码
        
        参数:
            img: 输入图像 [B, C, H, W]
            
        返回:
            mask: [B, 1, H, W] - 像素级掩码
            mask_token: [B, 3, H, W] - 掩码令牌
            p_x_full: [B, 1, H, W] - 像素级概率图
        """
        # 获取块嵌入
        B_input, C_input, H_input, W_input = img.shape
        x = self.patch_embed(img)  # [B, N, C]
        B, N, C = x.shape
        
        # 生成像素级掩码
        p_x, p_x_pixel, pixel_mask = self.get_mask(x, (H_input, W_input))
        
        # 处理掩码 - 已经是像素级的，直接添加通道维度
        mask = pixel_mask.unsqueeze(1).float().contiguous()
        
        # 处理概率图 - 已经是像素级的，直接添加通道维度
        p_x_full = p_x_pixel.unsqueeze(1).contiguous()
        
        # 创建掩码令牌
        mask_token = torch.zeros((B, 3, H_input, W_input)).to(img.device)
        
        return mask, mask_token, p_x_full

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # 生成基于正弦余弦函数的固定位置编码
    import numpy as np
    
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度使用sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度使用cos

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 添加批次维度 [1, n_position, d_hid]