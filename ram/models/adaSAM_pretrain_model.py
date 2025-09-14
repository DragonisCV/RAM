import torch
from collections import OrderedDict
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
from ram.utils import tensor2img
from ram.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from ram.utils.dist_util import master_only
from ram.archs import build_network
from ram.losses import build_loss
from ram.utils import get_root_logger, imwrite, tensor2img

@MODEL_REGISTRY.register()
class adaSAMPretrainModel(BaseModel):
    """MIM model for masked image modeling."""

    def __init__(self, opt):
        super(adaSAMPretrainModel, self).__init__(opt)
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.net_mask = build_network(opt["network_mask"])
        self.net_mask = self.model_to_device(self.net_mask)
        self.print_network(self.net_mask)
        if self.is_train:
            self.init_training_settings()
        else:
            self.load_pretrained_models()
       
    def init_training_settings(self):
        self.net_g.train()
        self.net_mask.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.init_ema_model()

        self.load_pretrained_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_functions()
       
    def load_pretrained_models(self):
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        
        load_path_mask = self.opt['path'].get('pretrain_network_mask', None)
        if load_path_mask is not None:
            param_key = self.opt['path'].get('param_key_mask', 'params')
            self.load_network(self.net_mask, load_path_mask, self.opt['path'].get('strict_load_mask', True), param_key)
    
    def init_ema_model(self):
        logger = get_root_logger()
        logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
        
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  
        self.net_g_ema.eval()
        

        self.net_mask_ema = build_network(self.opt['network_mask']).to(self.device)  
        load_path_mask = self.opt['path'].get('pretrain_network_mask', None)
        if load_path_mask is not None:
            param_key = self.opt['path'].get('param_key_mask', 'params')
            self.load_network(self.net_mask_ema, load_path_mask, self.opt['path'].get('strict_load_mask', True), param_key)
        else:
            self.mask_model_ema(0)  
        self.net_mask_ema.eval()

    def mask_model_ema(self, decay=0.999):
        net_mask = self.get_bare_model(self.net_mask)
        net_mask_params = dict(net_mask.named_parameters())
        net_mask_ema_params = dict(self.net_mask_ema.named_parameters())

        for k in net_mask_ema_params.keys():
            net_mask_ema_params[k].data.mul_(decay).add_(net_mask_params[k].data, alpha=1 - decay)

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_params = []
        for k, v in self.net_mask.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_mask"].pop("type")
        self.optimizer_mask = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_mask"]
        )

        self.optimizers.append(self.optimizer_mask)

    def setup_loss_functions(self):
        train_opt = self.opt['train']
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        self.cri_cls = build_loss(train_opt['cls_opt']).to(self.device) if train_opt.get('cls_opt') else None
        self.cri_cosine = build_loss(train_opt['cosine_opt']).to(self.device) if train_opt.get('cosine_opt') else None
        self.cri_high_freq = build_loss(train_opt['high_freq_opt']).to(self.device) if train_opt.get('high_freq_opt') else None
        self.cri_prob_pix = build_loss(train_opt['prob_pix_opt']).to(self.device) if train_opt.get('prob_pix_opt') else None
        if self.cri_pix is None and self.cri_perceptual is None and self.cri_cls is None and self.cri_cosine is None and self.cri_high_freq is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.gt_path = data['gt_path']
        self.lq = data.get('lq', None)
        if self.lq is not None:
            self.lq = self.lq.to(self.device)
        self.lq_path = data.get('lq_path', None)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_mask.zero_grad()

        self.mask, self.mask_token, self.p_x = self.net_mask(self.lq)
        self.output = self.net_g(self.lq, self.mask.detach(), self.mask_token.detach())

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt, self.mask)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_prob_pix: 
            l_prob_pix = self.cri_prob_pix(self.output.detach(), self.gt, self.mask, self.p_x)
            l_total += l_prob_pix
            loss_dict['l_prob_pix'] = l_prob_pix

        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_mask.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            self.mask_model_ema(decay=self.ema_decay)
    
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_mask_ema'):
            self.save_network([self.net_mask, self.net_mask_ema], 'net_mask', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_mask, 'net_mask', current_iter)
        self.save_training_state(epoch, current_iter)
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        
        if hasattr(self, 'mask'):
            mask_vis = self.mask.detach().cpu()
            out_dict['mask_img'] = mask_vis
            
            if hasattr(self, 'mask_token'):
                mask_applied = self.lq.detach().cpu() * (1 - mask_vis) + self.mask_token.detach().cpu() * mask_vis
                out_dict['mask_applied'] = mask_applied
        
        return out_dict
        

    @master_only
    def save_image(self, current_iter, img_name):
        visuals = self.get_current_visuals()
        
        train_dataset_opt = self.opt.get('datasets', {}).get('train', {})
        if 'mean' in train_dataset_opt and 'std' in train_dataset_opt:
            mean = torch.tensor(train_dataset_opt['mean'])
            std = torch.tensor(train_dataset_opt['std'])
            denorm_mean = -mean / std
            denorm_std = 1.0 / std
            for key in visuals.keys():
                if isinstance(visuals[key], torch.Tensor) and key != 'mask_img' and key != 'p_x':  
                    from torchvision.transforms.functional import normalize
                    normalize(visuals[key], denorm_mean, denorm_std, inplace=True)
        out_img = []
        if 'lq' in visuals:
            lq_img = tensor2img([visuals['lq']])
            out_img.append(lq_img)

        if 'mask_img' in visuals:
            try:
                mask_tensor = 1.0 - visuals['mask_img'] 
                
                if mask_tensor.max() > 1.0 or mask_tensor.min() < 0.0:
                    mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-6)
                
                try:
                    mask_img = tensor2img([mask_tensor], min_max=(0, 1))
                except TypeError:
                    mask_tensor = mask_tensor * 255.0
                    mask_img = tensor2img([mask_tensor])
                
                out_img.append(mask_img)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f"Error converting mask to image: {e}")
        
        if 'mask_applied' in visuals:
            try:
                mask_applied_img = tensor2img([visuals['mask_applied']])
                out_img.append(mask_applied_img)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f"Error converting mask_applied to image: {e}")
        
        if 'result' in visuals:
            result_img = tensor2img([visuals['result']])
            out_img.append(result_img)
        
        if 'gt' in visuals:
            gt_img = tensor2img([visuals['gt']])
            out_img.append(gt_img)
        
        sr_img = np.hstack(out_img)

        del self.lq
        del self.output
        torch.cuda.empty_cache()

        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'],
                                     f'{current_iter}_{img_name}.png')
        else:
            dataset_name = self.opt['datasets']['test']['name']
            if self.opt['val']['suffix']:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_{self.opt["val"]["suffix"]}.png')
            else:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'{img_name}_{self.opt["name"]}.png')
        
        imwrite(sr_img, save_img_path)