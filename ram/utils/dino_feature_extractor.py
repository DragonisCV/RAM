import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModel
import math
class DinoFeatureModule(nn.Module):
    def __init__(self, dino_model='dinov2_giant', pretrained_path=r'pretrained_model/facebookdinov2_giant',adaptive_mask = True,img_size = 128):
        super(DinoFeatureModule, self).__init__()
        
        # 使用 AutoModel 加载模型并转换为半精度
        self.dino = AutoModel.from_pretrained(
            pretrained_path,
            local_files_only=True,
            torch_dtype=torch.float16  # 设置为半精度
        )
        
        # 设置为评估模式
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # 验证模型是否确实被冻结
        frozen = all(not p.requires_grad for p in self.dino.parameters())
        assert frozen, "DINOv2 model parameters are not completely frozen!"
        
        # DINOv2-giant has a fixed embedding dimension of 1536
        self.shallow_dim = 1536  # Layer 13 features
        self.mid_dim = 1536      # Layer 26 features
        self.deep_dim = 1536     # Layer 39 features
        self.adaptive_mask = adaptive_mask
        self.mask_token = torch.zeros(1, 3, img_size, img_size)

        
    def get_dino_features(self, x):
        """
        获取DINO特征并重组为空间特征图
        x: [B, 3, H, W]
        """
        with torch.no_grad():  # DINO 特征提取过程不需要梯度
            outputs = self.dino(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # 获取输入图像的高宽比
            _, _, H, W = x.shape
            aspect_ratio = W / H
            
            # 选择特定层的特征
            shallow_feat1 = hidden_states[7]  
            shallow_feat2 = hidden_states[15]
            mid_feat1 = hidden_states[20]
            mid_feat2 = hidden_states[22]
            deep_feat1 = hidden_states[33]
            deep_feat2 = hidden_states[39]

            def reshape_features(feat):
                # 移除CLS token并重组为空间特征
                feat = feat[:, 1:, :]
                B, N, C = feat.shape
                
                # 根据原始图像的宽高比计算特征图的高度和宽度
                h = int(math.sqrt(N / aspect_ratio))
                w = int(N / h)
                
                # 添加检查确保尺寸正确  增加小的收益高
                if(aspect_ratio > 1): #w>h
                    if h * w > N:
                        h -= 1
                        w = N // h
                    if h * w < N:
                        h += 1
                        w = N // h
                else: #h>w
                    if h * w > N:
                        w -= 1
                        h = N // w
                    if h * w < N:
                        w += 1
                        h = N // w
                
                # 确保维度匹配
                assert h * w == N, f"Dimensions mismatch: {h}*{w} != {N}"
                
                # 重组为空间特征图
                feat = feat.reshape(B, h, w, C).permute(0, 3, 1, 2)
                return feat

            # 重组特征并转回单精度
            shallow_feat1 = reshape_features(shallow_feat1).float()
            mid_feat1 = reshape_features(mid_feat1).float()
            deep_feat1 = reshape_features(deep_feat1).float()
            shallow_feat2 = reshape_features(shallow_feat2).float()
            mid_feat2 = reshape_features(mid_feat2).float()
            deep_feat2 = reshape_features(deep_feat2).float()

            return shallow_feat1, mid_feat1, deep_feat1, shallow_feat2, mid_feat2, deep_feat2
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_size = 16
        mod_pad_h = (pad_size - h % pad_size) % pad_size
        mod_pad_w = (pad_size - w % pad_size) % pad_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img , mask=None, mask_tokens = None):
        if mask is not None:
            if self.adaptive_mask:
                mask = mask.repeat_interleave(1, 1).repeat_interleave(1, 2).contiguous()
            else:
                mask = mask.repeat_interleave(1, 1).repeat_interleave(1, 2).unsqueeze(1).contiguous() 
            if mask_tokens is None:
                mask_tokens = self.mask_token.expand(b,-1,-1,-1)

            inp_img = inp_img * mask + mask_tokens * (1-mask) #unmasked
        
        device = inp_img.device
        # 定义标准化参数
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


        denormalized_img = inp_img * std + mean
        denormalized_img = self.check_image_size(denormalized_img)
        h_denormalized, w_denormalized = denormalized_img.shape[2], denormalized_img.shape[3]
        # 计算目标高度和宽度，确保是14的倍数
        target_h = (h_denormalized // 8) * 14  
        target_w = (w_denormalized // 8) * 14  
        # 使用较短边作为 shortest_edge
        shortest_edge = min(target_h, target_w)
        processor = AutoImageProcessor.from_pretrained(
            r'pretrained_model/facebookdinov2_giant',
            local_files_only=True,
            do_rescale=False,
            do_center_crop=False,  # 禁用中心裁剪
            size={"shortest_edge": shortest_edge},
            use_fast=True
        )
        # logger = get_root_logger()
        inputs = processor(
            images=denormalized_img,  # 使用反标准化后的图像
            return_tensors="pt"
        ).to(device)
            
        # 获取原始DINO特征
        shallow_feat1, mid_feat1, deep_feat1, shallow_feat2, mid_feat2, deep_feat2 = self.get_dino_features(inputs['pixel_values'])
        
        dino_features = {
            'shallow_feat1': shallow_feat1,
            'mid_feat1': mid_feat1,
            'deep_feat1': deep_feat1,
            'shallow_feat2': shallow_feat2,
            'mid_feat2': mid_feat2,
            'deep_feat2': deep_feat2
        }
        
        return dino_features