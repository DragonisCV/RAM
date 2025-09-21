import os
from ram.data.base.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.data_util import paired_paths_from_folder


@DATASET_REGISTRY.register()
class LOLv1Dataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None,enlarge_ratio=1):
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
        self.paths = self.paths * enlarge_ratio  # Following prior work, we perform data augmentation.
        
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

@DATASET_REGISTRY.register()
class LOLv2Dataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None,enlarge_ratio=1):
        super(LOLv2Dataset, self).__init__(opt)
        self.folder = dataroot or opt['dataroot']
        self.augmentator = augmentator
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.paths = self._get_image_paths()
        self.paths = self.paths * enlarge_ratio
        
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

    def _get_image_paths(self):
        real_lq_folder = os.path.join(self.folder, 'Real_captured/Train/Low/')
        real_gt_folder = os.path.join(self.folder, 'Real_captured/Train/Normal/')
        syn_lq_folder = os.path.join(self.folder, 'Synthetic/Train/Low/')
        syn_gt_folder = os.path.join(self.folder, 'Synthetic/Train/Normal/')

        real_lq_names = os.listdir(real_lq_folder)
        real_gt_names = [n.replace('low', 'normal') for n in real_lq_names]

        real_paths = [
            {
                'lq_path': os.path.join(real_lq_folder, lq_name),
                'gt_path': os.path.join(real_gt_folder, gt_name)
            }
            for lq_name, gt_name in zip(real_lq_names, real_gt_names)
        ]

        syn_paths = paired_paths_from_folder(
            [syn_lq_folder, syn_gt_folder], 
            ['lq', 'gt'], 
            self.filename_tmpl
        )

        return real_paths + syn_paths