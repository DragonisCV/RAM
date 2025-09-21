import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModel
import math
class DinoFeatureModule(nn.Module):
    def __init__(self, dino_model='dinov2_giant', pretrained_path=r'pretrained_model/facebookdinov2_giant',img_size = 128):
        super(DinoFeatureModule, self).__init__()
        
        self.dino = AutoModel.from_pretrained(
            pretrained_path,
            local_files_only=False,
            torch_dtype=torch.float16  
        )
        
      
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
        
       
        frozen = all(not p.requires_grad for p in self.dino.parameters())
        assert frozen, "DINOv2 model parameters are not completely frozen!"
        
        
        self.shallow_dim = 1536  
        self.mid_dim = 1536      
        self.deep_dim = 1536     
        
    def get_dino_features(self, x):
        with torch.no_grad(): 
            outputs = self.dino(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
           
            _, _, H, W = x.shape
            aspect_ratio = W / H
          
            shallow_feat1 = hidden_states[7]  
            shallow_feat2 = hidden_states[15]
            mid_feat1 = hidden_states[20]
            mid_feat2 = hidden_states[22]
            deep_feat1 = hidden_states[33]
            deep_feat2 = hidden_states[39]

            def reshape_features(feat):
                feat = feat[:, 1:, :]
                B, N, C = feat.shape
                
                h = int(math.sqrt(N / aspect_ratio))
                w = int(N / h)
                
               
                if(aspect_ratio > 1):
                    if h * w > N:
                        h -= 1
                        w = N // h
                    if h * w < N:
                        h += 1
                        w = N // h
                else: 
                    if h * w > N:
                        w -= 1
                        h = N // w
                    if h * w < N:
                        w += 1
                        h = N // w
                
                
                assert h * w == N, f"Dimensions mismatch: {h}*{w} != {N}"
                
                
                feat = feat.reshape(B, h, w, C).permute(0, 3, 1, 2)
                return feat

          
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

    def forward(self, inp_img):
        
        device = inp_img.device
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


        denormalized_img = inp_img * std + mean
        denormalized_img = self.check_image_size(denormalized_img)
        h_denormalized, w_denormalized = denormalized_img.shape[2], denormalized_img.shape[3]
        # To ensure minimal changes and maintain code generality, the image size is directly scaled here to guarantee spatial alignment.

        target_h = (h_denormalized // 8) * 14  
        target_w = (w_denormalized // 8) * 14  
       
        shortest_edge = min(target_h, target_w)
        processor = AutoImageProcessor.from_pretrained(
            r'pretrained_model/facebookdinov2_giant',
            local_files_only=False,
            do_rescale=False,
            do_center_crop=False,  
            use_fast=True,
            size={"shortest_edge": shortest_edge}
        )
        
        inputs = processor(
            images=denormalized_img,  
            return_tensors="pt"
        ).to(device)
            
       
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