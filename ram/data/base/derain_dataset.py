import os
from os import path as osp
import torch
from ram.data.base.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.data_util import paired_paths_from_folder
@DATASET_REGISTRY.register()
class Rain100LTrainDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None, enlarge_ratio=1):
        super(Rain100LTrainDataset, self).__init__(opt)
        self.folder = dataroot or opt['dataroot']
        self.augmentator = augmentator
        self.paths = self._get_image_paths()
        self.paths = self.paths * enlarge_ratio  # Following prior work, we perform data augmentation.

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

        img_gt, img_lq = self._process_images(img_gt, img_lq)
        
        return_dict = {
            'lq': img_lq,
            'gt': img_gt,
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




@DATASET_REGISTRY.register()
class Rain13kDataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None,enlarge_ratio=1):
        super(Rain13kDataset, self).__init__(opt)
        self.gt_folder = gt_path or opt['dataroot_gt']
        self.lq_folder = lq_path or opt['dataroot_lq']
        self.augmentator = augmentator
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], 
            ['lq', 'gt'],
            self.opt.get('filename_tmpl', '{}')
        )
        self.paths = self.paths * enlarge_ratio

    def __getitem__(self, index):
        self._init_file_client()
        scale = self.opt['scale']

        # 加载GT和LQ图像
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_gt = self._load_image(gt_path, 'gt')
        img_lq = self._load_image(lq_path, 'lq')

        # 数据增强
        if self.opt['phase'] == 'train':
            img_gt, img_lq = self._train_augmentation(img_gt, img_lq, scale, gt_path)
        else:
            img_gt, img_lq = self._test_processing(img_gt, img_lq, scale)

        # 应用额外的增强器
        if self.augmentator:
            img_lq = self.augmentator(img_lq)

        # 图像处理：BGR到RGB，HWC到CHW，numpy到tensor，标准化
        img_gt, img_lq = self._process_images(img_gt, img_lq)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
