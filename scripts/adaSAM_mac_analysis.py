from os import path as osp
from ram.utils.options import parse_options
import torch
from scripts.analysis_utils import BaseAnalysis,attr_grad
import numpy as np
from tqdm import tqdm
from ram.archs import build_network
from ram.archs.AdaSAM_arch import AdaptiveMaskPixGenerator

class Ada_MACAnalysis(BaseAnalysis):
    def __init__(self, opt):
        super().__init__(opt)
        self.mask_generator = self._load_mask_generator(opt)
        self.mask_generator.eval()
        
    def _load_mask_generator(self, opt):
        mask_opt = opt.get('network_mask', {})
        mask_generator = build_network(mask_opt).to(self.device)
        mask_path_opt = opt.get('net_mask_path', {})
        mask_path = mask_path_opt.get('path', None)
        self.load_network(
            mask_generator, 
            mask_path, 
            strict=mask_path_opt.get('strict_load', True),
            param_key=mask_path_opt.get('param_key', 'params')
        )
        mask_generator.eval()
        return mask_generator
        
    def analyze(self):
        total_filter_mac = [0.0] * len(self.hook_list)
        for test_loader in self.test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            num_samples = self.opt.get('num_samples',10) 
            print(f'Analyzing {test_set_name}..\n')
            pbar = tqdm(total=num_samples, desc='')
            for idx, val_data in enumerate(test_loader):
                if idx >= num_samples:
                    break
                tensor_lq = val_data['lq'].to(self.device)
                imgname = osp.basename(val_data['lq_path'][0])
                tensor_base = torch.zeros_like(tensor_lq)
                layer_conductance = self._mask_attribute_conductance(tensor_base, tensor_lq)
                total_filter_mac = [a + b for a, b in zip(total_filter_mac, layer_conductance)]
                pbar.set_description(f'Read {imgname}')
                pbar.update(1)
        self._save_results(total_filter_mac, 'mac')

    def _mask_attribute_conductance(self, base_img, final_img):
        total_step = self.opt['total_step']

        with torch.no_grad():
            _, _, p_x_full = self.mask_generator(final_img)
        p_x_flat = p_x_full.flatten()
        order_array = torch.argsort(p_x_flat).cpu().numpy()

        start_ratio = self.opt['pretrained_ratio']
        all_hook_layer_conductance = [0.0] * len(self.hook_list)
        last_hook_layer_output = []

        for step in range(total_step):
            alpha = 1 - start_ratio + start_ratio * step / total_step
            interpolated_img = self._get_interpolated_img_from_mask_attribute_path(base_img, final_img, alpha, order_array).to(self.device)
            self.model.zero_grad()
            interpolated_output = self.model(interpolated_img,None,None)
            
            if isinstance(interpolated_output, tuple):
                interpolated_output = interpolated_output[0]
            
            loss = attr_grad(interpolated_output, reduce='sum')
            loss.backward()
            now_hook_layer_output = []
            for hook in self.hook_list:
                if hasattr(hook, 'output') and hook.output is not None:
                    now_hook_layer_output.append(hook.output.detach())
                else:
                    now_hook_layer_output.append(None)

            if step > 0:
                dfdy = []
                approx_dydx = []
                for i, hook in enumerate(self.hook_list):
                    if hasattr(hook, 'grad') and hook.grad is not None and now_hook_layer_output[i] is not None:
                        dfdy.append(hook.grad.detach())
                        approx_dydx.append(now_hook_layer_output[i] - last_hook_layer_output[i])
                    else:
                        dfdy.append(torch.zeros_like(last_hook_layer_output[i]) if last_hook_layer_output[i] is not None else None)
                        approx_dydx.append(torch.zeros_like(last_hook_layer_output[i]) if last_hook_layer_output[i] is not None else None)

                for i, (df, dy) in enumerate(zip(dfdy, approx_dydx)):
                    if df is not None and dy is not None:
                        all_hook_layer_conductance[i] += df * dy

            last_hook_layer_output = now_hook_layer_output
        return [torch.mean(torch.abs(cond) if isinstance(cond, torch.Tensor) else torch.tensor(0.0)).detach().cpu().numpy() for cond in all_hook_layer_conductance]

def main():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, _ = parse_options(root_path, is_train=False)

    analysis = Ada_MACAnalysis(opt)
    analysis.analyze()

if __name__ == '__main__':
    main()