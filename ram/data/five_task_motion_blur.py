import os
from os import path as osp
import torch
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.data_util import paired_paths_from_folder

@DATASET_REGISTRY.register()
class DeblurGoProDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None):
        super(DeblurGoProDataset, self).__init__(opt)
        self.folder = dataroot or opt['dataroot']
        self.dino_folder = opt.get('dataroot_dino', None)  # DINO特征路径，可选
        self.augmentator = augmentator
        self.paths = self._get_image_paths()
        self.paths = self.paths * 5 # Following prior work, we perform data augmentation.
        if self.opt.get('limit_ratio'):
            self._limit_dataset()

    def _get_image_paths(self):
        paths = []
        for dataset in os.listdir(self.folder):
            lq_folder = os.path.join(self.folder, dataset, 'blur')
            gt_folder = os.path.join(self.folder, dataset, 'sharp')
            paths.extend(paired_paths_from_folder(
                [lq_folder, gt_folder], 
                ['lq', 'gt'], 
                self.opt.get('filename_tmpl', '{}')
            ))
        return paths


    def __getitem__(self, index):
        self._init_file_client()
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_gt = self._load_image(gt_path, 'gt')
        img_lq = self._load_image(lq_path, 'lq')

        if self.opt['phase'] == 'train':
            img_gt, img_lq = self._train_augmentation(img_gt, img_lq, scale, gt_path)
        else:
            img_gt, img_lq = self._test_processing(img_gt, img_lq, scale)

        if self.augmentator:
            img_lq = self.augmentator(img_lq)

        processed = self._process_images(img_gt, img_lq)
        
        return_dict = {
            'lq': processed['lq'],
            'gt': processed['gt'],
            'lq_path': lq_path,
            'gt_path': gt_path
        }

        return return_dict

    def _test_preprocessing(self, img):
        if self.opt.get('test_crop', False):
            return center_crop(img, self.opt['gt_size'])
        return img

    def __len__(self):
        return len(self.paths)