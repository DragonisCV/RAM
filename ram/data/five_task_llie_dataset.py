from ram.data.utils.data_util import paired_paths_from_folder
from ram.utils.registry import DATASET_REGISTRY
from ram.data.base_dataset import BaseDataset
from os import path as osp
import torch

@DATASET_REGISTRY.register()
class LOLv1Dataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None):
        super(LOLv1Dataset, self).__init__(opt)
        self.gt_folder = gt_path or opt['dataroot_gt']
        self.lq_folder = lq_path or opt['dataroot_lq']
        self.dino_folder = opt.get('dataroot_dino', None)  
        self.augmentator = augmentator
        
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], 
            ['lq', 'gt'],
            self.opt.get('filename_tmpl', '{}')
        )
        self.paths = self.paths * 20   # Following prior work, we perform data augmentation.
        
        if self.opt.get('limit_ratio'):
            self._limit_dataset()

    def __getitem__(self, index):
        self._init_file_client()
        scale = self.opt['scale']

        img_gt = self._load_image(self.paths[index]['gt_path'], 'gt')
        img_lq = self._load_image(self.paths[index]['lq_path'], 'lq')

        if self.opt['phase'] == 'train':
            img_gt, img_lq = self._train_augmentation(img_gt, img_lq, scale)
        else:
            img_gt, img_lq = self._test_processing(img_gt, img_lq, scale)

        if self.augmentator is not None:
            img_lq = self.augmentator(img_lq)

        img_gt, img_lq = self._process_images(img_gt, img_lq)

        return {
            'lq': img_lq, 
            'gt': img_gt, 
            'lq_path': self.paths[index]['lq_path'], 
            'gt_path': self.paths[index]['gt_path']
        }

    def __len__(self):
        return len(self.paths)
