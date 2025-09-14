import logging
import torch
from os import path as osp
from torch import distributed as dist

from ram.data import build_dataloader, build_dataset
from ram.models import build_model
from ram.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from ram.utils.options import dict2str, parse_options

def init_dist_fn(launcher, backend='nccl', **kwargs):
    """Initialize distributed training."""
    if launcher == 'pytorch':
        kwargs.pop('port', None)  
        init_fn = dist.init_process_group
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')
    
    init_fn(backend=backend, **kwargs)

def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    
    if opt['dist']:
        if not dist.is_initialized():
            init_dist_fn(opt['launcher'], **opt['dist_params'])
        
        opt['rank'] = torch.distributed.get_rank()
        opt['world_size'] = torch.distributed.get_world_size()
    
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='ram', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['phase'] = 'test'  
        dataset_opt['gt_size'] = opt['gt_size'] 
        test_set = build_dataset(dataset_opt)
        
        if opt['dist']:
            sampler = torch.utils.data.distributed.DistributedSampler(
                test_set,
                num_replicas=opt['world_size'],
                rank=opt['rank'],
                shuffle=False
            )
            
            dataset_opt['batch_size_per_gpu'] = dataset_opt.get('batch_size', 1)
            dataset_opt['num_worker_per_gpu'] = 2
            dataset_opt['pin_memory'] = True
            dataset_opt['phase'] = 'train' 
        else:
            sampler = None
            
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=sampler)
        test_loaders.append(test_loader)
        dataset_opt['phase'] = 'test'

    model = build_model(opt)
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader, 
            current_iter=opt['name'], 
            tb_logger=None, 
            save_img=opt['val']['save_img'],
            test_num=opt.get('test_num',-1),
            save_num=opt.get('save_num',-1))

    if opt['dist']:
        logger.info(f'Rank {opt["rank"]}: Dataset size = {len(test_set)}')

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)