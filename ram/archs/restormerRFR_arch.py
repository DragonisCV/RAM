# RAM++: Robust Representation Learning via Adaptive Mask for All-in-One Image Restoration
# Zilong Zhang, Chujie Qin, Chunle Guo, Yong Zhang, Chao Xue, Ming-Ming Cheng and Chongyi Li
# https://arxiv.org/abs/2509.12039

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange

from ram.utils.registry import ARCH_REGISTRY


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,finetune_type=None):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
                
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(dim),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,finetune_type=None):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias,finetune_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return F.softmax(x / torch.clamp(self.temperature, min=1e-8), dim=1)

class DinoFeatureFusion(nn.Module):
    def __init__(self, dino_dim=1536):
        super(DinoFeatureFusion, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.gate_network = nn.Sequential(
            nn.Linear(dino_dim * 2, dino_dim),
            nn.PReLU(dino_dim),
            nn.Linear(dino_dim, 512),
            nn.PReLU(512),
            nn.Linear(512, 2),
            TemperatureSoftmax(self.temperature) 
        )

    def forward(self, dino_feat1, dino_feat2):
        pooled_feat1 = self.global_pool(dino_feat1).squeeze(-1).squeeze(-1) 
        pooled_feat2 = self.global_pool(dino_feat2).squeeze(-1).squeeze(-1) 
        pooled_features = torch.cat([pooled_feat1, pooled_feat2], dim=1)  

        weights = self.gate_network(pooled_features)  
        weighted_feat1 = dino_feat1 * weights[:, 0:1].view(-1, 1, 1, 1)
        weighted_feat2 = dino_feat2 * weights[:, 1:2].view(-1, 1, 1, 1)

        fused_feat = weighted_feat1 + weighted_feat2  
        return fused_feat
        
        




class DRAdaptation(nn.Module):
    def __init__(self, dino_dim=1536, restore_dim=48, scale_factor=14, size=128):      
        super(DRAdaptation, self).__init__()
        self.size = size  
        self.restore_dim = restore_dim
        self.adaptation = nn.Sequential(
            nn.Conv2d(dino_dim, restore_dim*16, kernel_size=3, padding=1), #768
            nn.PReLU(restore_dim*16),
            nn.Conv2d(restore_dim*16, restore_dim*8, kernel_size=1),#384
        )

    def forward(self, dino_feat, restore_feat):
        B, C, H, W = restore_feat.shape
        
        adapted_dino = self.adaptation(dino_feat)  

        return adapted_dino



##########################################################################
##---------- D-R Fusion -----------------------
class DinoRestoreFeatureFusion(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DinoRestoreFeatureFusion, self).__init__()
        self.reduce_chan = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
    def forward(self, dino_feat, restore_feat):
        x_fusion = self.reduce_chan(torch.cat([dino_feat, restore_feat], dim=1))
        res = x_fusion + restore_feat
        return res
        
    
##---------- restormerRFR -----------------------


@ARCH_REGISTRY.register()
class RestormerRFR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
        finetune_type = None,
        img_size = 128
    ):

        super(RestormerRFR, self).__init__()
        
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.mask_token = torch.zeros(1, 3, img_size, img_size)
        
        self.dr_adaptation1 = DRAdaptation(dino_dim=1536, restore_dim=48, scale_factor=14, size=128)
        self.dr_adaptation2 = DRAdaptation(dino_dim=1536, restore_dim=48, scale_factor=14, size=128)
        self.dr_adaptation3 = DRAdaptation(dino_dim=1536, restore_dim=48, scale_factor=14, size=128)
        self.dr_fusion1 = DinoRestoreFeatureFusion(dim=int(dim*2**3), num_heads=heads[3], bias=bias) 
        self.dr_fusion2 = DinoRestoreFeatureFusion(dim=int(dim*2**2), num_heads=heads[2], bias=bias)
        self.dr_fusion3 = DinoRestoreFeatureFusion(dim=int(dim*2**1), num_heads=heads[1], bias=bias)
        self.up_4_3_dino1 = Upsample(int(dim*2**3))
        self.up_4_3_dino2 = Upsample(int(dim*2**3))
        self.up_3_2_dino = Upsample(int(dim*2**2))
        self.dino_fusion_shallow = DinoFeatureFusion(dino_dim=1536)
        self.dino_fusion_mid = DinoFeatureFusion(dino_dim=1536)
        self.dino_fusion_deep = DinoFeatureFusion(dino_dim=1536)



        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[0]-1 else None) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) 
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[1]-1 else None) for i in range(num_blocks[1])])       
        self.down2_3 = Downsample(int(dim*2**1)) 
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[2]-1 else None) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2)) 
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[3]-1 else None) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) 
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[2]-1 else None) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[1]-1 else None) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1)) 
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[0]-1 else None) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_refinement_blocks-1 else None) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_size = 16
        mod_pad_h = (pad_size - h % pad_size) % pad_size
        mod_pad_w = (pad_size - w % pad_size) % pad_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, inp_img, dino_features =None ):
        b,c,h,w = inp_img.shape

        shallow_feat1, mid_feat1, deep_feat1, shallow_feat2, mid_feat2, deep_feat2 = dino_features.values()
        inp_img = self.check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        

        latent = self.latent(inp_enc_level4)  
        

       
        
        shallow_feat = self.dino_fusion_shallow(shallow_feat1, shallow_feat2)
        mid_feat = self.dino_fusion_mid(mid_feat1, mid_feat2)
        deep_feat = self.dino_fusion_deep(deep_feat1, deep_feat2)

        shallow_feat = self.dr_adaptation1(shallow_feat, latent) 
        mid_feat = self.dr_adaptation2(mid_feat, latent)
        deep_feat = self.dr_adaptation3(deep_feat, latent)

        latent = self.dr_fusion1(dino_feat=deep_feat, restore_feat=latent) 
        shallow_feat = self.up_4_3_dino1(shallow_feat)
        mid_feat = self.up_4_3_dino2(mid_feat)

        inp_dec_level3 = self.up4_3(latent) 
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        out_dec_level3 = self.dr_fusion2(dino_feat=mid_feat, restore_feat=out_dec_level3) 
        shallow_feat = self.up_3_2_dino(shallow_feat)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)  

        out_dec_level2 = self.dr_fusion3(dino_feat=shallow_feat, restore_feat=out_dec_level2) 


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)


        return out_dec_level1[:,:,:h,:w]