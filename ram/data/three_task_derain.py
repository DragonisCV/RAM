import os
from os import path as osp
import torch
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Rain100LTrainDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None):
        super(Rain100LTrainDataset, self).__init__(opt)
        self.folder = dataroot or opt['dataroot']
        self.augmentator = augmentator
        self.paths = self._get_image_paths()
        self.paths = self.paths * 120  # Following prior work, we perform data augmentation.
        if self.opt.get('limit_ratio'):
            self._limit_dataset()

    def _get_image_paths(self):
        paths = []
        lq_folder = self.folder
        paths = []
        
        for filename in os.listdir(lq_folder):
            if filename.startswith('norain-'):
                rain_image = 'rain-' + filename[7:] 
                norain_image_path = os.path.join(lq_folder, filename)
                rain_image_path = os.path.join(lq_folder, rain_image)
                if os.path.exists(rain_image_path):
                    paths.append({'lq_path': rain_image_path, 'gt_path': norain_image_path})
        
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
