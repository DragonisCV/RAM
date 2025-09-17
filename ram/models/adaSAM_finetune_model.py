import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import torch.distributed as dist

from ram.archs import build_network
from ram.losses import build_loss
from ram.metrics import calculate_metric, calculate_psnr, calculate_ssim
from ram.utils import get_root_logger, imwrite, tensor2img
from ram.utils.registry import MODEL_REGISTRY
from ram.utils.dist_util import master_only
from .base_model import BaseModel
from ram.utils.dino_feature_extractor import DinoFeatureModule


@MODEL_REGISTRY.register()
class adaSAMFinetuneModel(BaseModel):
    """MIM Stage 2 model for image restoration."""

    def __init__(self, opt):
        super(adaSAMFinetuneModel, self).__init__(opt)
        
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings()
        else:
            self.load_pretrained_models()

        self.metric_results = {}
        if 'val' in opt and 'metrics' in opt['val']:
            for metric in opt['val']['metrics'].keys():
                self.metric_results[metric] = 0.0
                
        self.best_metric_results = {}
        if 'val' in opt and 'metrics' in opt['val']:
            for key, dataset_opt in opt['datasets'].items():
                dataset_name = dataset_opt['name'] 
                self.best_metric_results[dataset_name] = dict()
                for metric in opt['val']['metrics'].keys():
                    self.best_metric_results[dataset_name][metric] = dict()
                    self.best_metric_results[dataset_name][metric]['val'] = -float('inf')
                    self.best_metric_results[dataset_name][metric]['iter'] = -1
                    self.best_metric_results[dataset_name][metric]['better'] = 'higher'
        
        self.dino_feature_extractor = DinoFeatureModule().to(self.device)

    def load_pretrained_models(self):
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            self.net_g.eval()
            if self.opt['network_g'].get('finetune_type') == 'cond':
                logger = get_root_logger()
                logger.info('Start setting up finetuning parameters...')
                
                filter_name_list = np.loadtxt(self.opt['finetune_block']['filter_txt_path'], dtype=str)
                filter_ratio = self.opt['finetune_block']['filter_ratio']
                logger.info(f'Loading filter list from {self.opt["finetune_block"]["filter_txt_path"]}')
                logger.info(f'Filter ratio set to: {filter_ratio}')
                
                finetune_name_list = filter_name_list[:int(len(filter_name_list)*filter_ratio)]
                finetune_name_list = ['module.' + name for name in finetune_name_list]
                logger.info(f'Will finetune {len(finetune_name_list)} modules')
                
                for name,param in self.net_g.named_parameters():
                    layer_name = '.'.join(name.split('.')[:-1])
                    if layer_name in finetune_name_list:
                        param.requires_grad = True
                        logger.info(f'Module {layer_name} will be finetuned')
                    else:
                        param.requires_grad = False
                        
                logger.info('Finetuning parameter setup completed')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.init_ema_model()

        self.load_pretrained_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_loss_functions()
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        logger = get_root_logger()
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
                logger.warning(f'Params {k} will be optimized.')
            
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
   
    def setup_loss_functions(self):
        train_opt = self.opt['train']
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None

        if self.cri_pix is None and self.cri_perceptual is None:
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
        self.dino_features = self.dino_feature_extractor(self.lq)
        deep_feat1 = self.dino_features['deep_feat1']
        self.dino_features = {key: deep_feat1 for key in self.dino_features}
        self.output = self.net_g(self.lq, self.dino_features)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.dino_features = self.dino_feature_extractor(self.lq)
            self.output = self.net_g(self.lq, self.dino_features)

        self.net_g.train()
    
   
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        normalize_opt = self.opt['val'].get('normalize', None)
        
        rank = self.opt.get('rank', 0)
        world_size = self.opt.get('world_size', 1)

        image_metrics = {}
        
        if normalize_opt:
            mean = -1 * torch.tensor(normalize_opt['mean']) / torch.tensor(normalize_opt['std'])
            std = 1 / torch.tensor(normalize_opt['std'])
        if self.opt['dist']:
            torch.distributed.barrier()
        if use_pbar and rank == 0:
            total_images = len(dataloader)
            pbar = tqdm(
                total=total_images, 
                unit='batch',
                desc=f'Testing {dataset_name}'
            )

        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = torch.zeros(len(self.opt['val']['metrics']) + 1, dtype=torch.float32, device='cuda')
            total_samples = 0

        for idx, val_data in enumerate(dataloader):
            if idx >= test_num > 0:
                break

            self.feed_data(val_data)
            
            with torch.no_grad():
                self.test()
                torch.cuda.synchronize()

            visuals = self.get_current_visuals()
            sr_imgs = []
            gt_imgs = []
            img_names = [] 
            for i in range(visuals['result'].size(0)):
                if normalize_opt:
                    normalize(visuals['result'][i:i+1], mean, std, inplace=True)
                    normalize(visuals['gt'][i:i+1], mean, std, inplace=True)
                    normalize(visuals['lq'][i:i+1], mean, std, inplace=True)

                sr_img = tensor2img([visuals['result'][i:i+1]])
                gt_img = tensor2img([visuals['gt'][i:i+1]])
                lq_img = tensor2img([visuals['lq'][i:i+1]])
                
                sr_imgs.append(sr_img)
                gt_imgs.append(gt_img)
                img_names.append(osp.splitext(osp.basename(val_data['lq_path'][i]))[0])
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][i]))[0]
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                               f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                   f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                   f'{img_name}_{self.opt["name"]}.png')
                    imwrite(np.hstack((lq_img, sr_img, gt_img)), save_img_path)

            if with_metrics:
                for img_idx, (sr_img, gt_img, img_name) in enumerate(zip(sr_imgs, gt_imgs, img_names)):
                    current_metrics = {}
                    for metric, opt in self.opt['val']['metrics'].items():
                        metric_type = opt.pop('type')
                        metric_opts = dict(opt)
                        
                        if metric_type == 'calculate_psnr':
                            value = calculate_psnr(sr_img, gt_img, **metric_opts)
                        elif metric_type == 'calculate_ssim':
                            value = calculate_ssim(sr_img, gt_img, **metric_opts)
                        else:
                            raise NotImplementedError(f'Metric type {metric_type} not recognized.')
                        
                        current_metrics[metric] = value
                        opt['type'] = metric_type
                        
                    relative_path = val_data['lq_path'][img_idx]
                    for _, dataset_opt in self.opt['datasets'].items():
                        if dataset_opt.get('dataroot'): 
                            dataroot = dataset_opt['dataroot']
                            if relative_path.startswith(dataroot):
                                relative_path = relative_path.replace(dataroot + '/', '')
                                break
                        elif dataset_opt.get('dataroot_lq'): 
                            dataroot = dataset_opt['dataroot_lq']
                            if relative_path.startswith(dataroot):
                                relative_path = relative_path.replace(dataroot + '/', '')
                                break
                    
                    image_metrics[f"{rank}_{relative_path}"] = current_metrics
                    
                    metric_data[-1] += 1
            if rank == 0 and use_pbar:
                pbar.update(1)
                pbar.set_description(f'Testing {dataset_name} Batch: {idx+1}/{total_images}')

            del self.lq
            del self.output
            torch.cuda.empty_cache()

        if self.opt['dist']:
            torch.distributed.barrier()

        if rank == 0 and use_pbar:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                all_image_metrics = [None for _ in range(world_size)]
                dist.all_gather_object(all_image_metrics, image_metrics)
                
                if rank == 0:
                    combined_metrics = {}
                    total_psnr = 0.0
                    total_ssim = 0.0
                    
                    processed_paths = set()
                    duplicates = set()

                    path_metrics = {}
                    
                    for gpu_metrics in all_image_metrics:
                        for img_name, metrics in gpu_metrics.items():
                            rank_id, full_path = img_name.split('_', 1)
                            if full_path in processed_paths:
                                duplicates.add(full_path)
                            processed_paths.add(full_path)
                            
                            if full_path not in path_metrics:
                                path_metrics[full_path] = []
                            path_metrics[full_path].append((img_name, metrics))
                    
                    processed_paths.clear()
                    for full_path, metrics_list in path_metrics.items():
                        avg_psnr = sum(m['psnr'] for _, m in metrics_list) / len(metrics_list)
                        avg_ssim = sum(m['ssim'] for _, m in metrics_list) / len(metrics_list)
                        
                        img_name = metrics_list[0][0]
                        combined_metrics[img_name] = {
                            'psnr': avg_psnr,
                            'ssim': avg_ssim
                        }
                        
                        total_psnr += avg_psnr
                        total_ssim += avg_ssim
                    
                    contributions = {}
                    for key in combined_metrics.keys():
                        rank_id = key.split('_')[0]
                        if rank_id not in contributions:
                            contributions[rank_id] = 0
                        contributions[rank_id] += 1
                    
                    unique_images = len(path_metrics)
                    self.metric_results['psnr'] = total_psnr / unique_images
                    self.metric_results['ssim'] = total_ssim / unique_images
                    
                    logger = get_root_logger()
                    logger.info(f'Total images processed: {len(processed_paths)}')
                    logger.info(f'Duplicate images found: {len(duplicates)}')
                    logger.info(f'Unique images used for metrics: {unique_images}')
                    logger.info(f'Per-GPU contributions: {contributions}')
                    
                    save_path = osp.join(self.opt['path']['visualization'], 
                                       f'metrics_{dataset_name}_{current_iter}.txt')
                    with open(save_path, 'w') as f:
                        f.write(f"Total images processed: {len(processed_paths)}\n")
                        f.write(f"Duplicate images found: {len(duplicates)}\n")
                        f.write(f"Unique images used for metrics: {unique_images}\n\n")
                        
                        for img_name, metrics in sorted(combined_metrics.items()):
                            rank_id, full_path = img_name.split('_', 1)
                            f.write(f"Image: {img_name}\n")
                            if full_path in duplicates:
                                f.write("*** DUPLICATE IMAGE ***\n")
                                f.write(f"Average of {len(path_metrics[full_path])} instances:\n")
                            for metric_name, value in metrics.items():
                                f.write(f"{metric_name}: {value:.4f}\n")
                            f.write("\n")

            if rank == 0:
                self._update_best_metric_results(dataset_name, current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, test_num=-1, save_num=-1):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        normalize_opt = self.opt['val'].get('normalize', None)
        
        if normalize_opt:
            mean = -1 * torch.tensor(normalize_opt['mean']) / torch.tensor(normalize_opt['std'])
            std = 1 / torch.tensor(normalize_opt['std'])

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if idx >= test_num > 0:
                break
            
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            if normalize_opt:
                normalize(visuals['result'], mean, std, inplace=True)
                normalize(visuals['gt'], mean, std, inplace=True)
                normalize(visuals['lq'], mean, std, inplace=True)

            sr_img = tensor2img([visuals['result']])
            gt_img = tensor2img([visuals['gt']])
            lq_img = tensor2img([visuals['lq']])

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                           f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                               f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                               f'{img_name}_{self.opt["name"]}.png')
                imwrite(np.hstack((lq_img, sr_img, gt_img)), save_img_path)

            if with_metrics:
                for metric in self.opt['val']['metrics'].keys():
                    metric_type = self.opt['val']['metrics'][metric].pop('type')
                    metric_opts = dict(self.opt['val']['metrics'][metric])
                    
                    if metric_type == 'calculate_psnr':
                        value = calculate_psnr(sr_img, gt_img, **metric_opts)
                    elif metric_type == 'calculate_ssim':
                        value = calculate_ssim(sr_img, gt_img, **metric_opts)
                    else:
                        raise NotImplementedError(f'Metric type {metric_type} not recognized.')
                        
                    self.metric_results[metric] += value
                    self.opt['val']['metrics'][metric]['type'] = metric_type

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

            del self.lq
            del self.output
            torch.cuda.empty_cache()

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                          f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _update_best_metric_results(self, dataset_name, current_iter):
        """Update the best metric results."""
        if hasattr(self, 'best_metric_results'):
            for metric in self.metric_results.keys():
                current_value = self.metric_results[metric]
                if current_value > self.best_metric_results[dataset_name][metric]['val']:
                    self.best_metric_results[dataset_name][metric]['val'] = current_value
                    self.best_metric_results[dataset_name][metric]['iter'] = current_iter

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
        
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)