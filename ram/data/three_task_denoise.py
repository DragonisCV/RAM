import os
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.transforms import augment, random_crop, center_crop, ensure_min_size
from ram.utils import imfrombytes
from ram.data.utils.online_util import parse_degradations,AugmentatorHub


@DATASET_REGISTRY.register()
class LowCostNoiseDataset(BaseDataset):
    def __init__(self, opt, dataroot=None, augmentator=None):
        super(LowCostNoiseDataset, self).__init__(opt)
        self.gt_folder = dataroot or opt['dataroot_gt']
        self.paths = self._get_image_paths()
        self.paths = self.paths * 3 # Following prior work, we perform data augmentation.
        self.augmentator = self._init_augmentator(augmentator)
        if self.opt.get('limit_ratio'):
            self._limit_dataset()

    def _get_image_paths(self):
        """获取所有图像路径"""
        image_paths = []
        
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in os.listdir(self.gt_folder)):
            image_paths.extend([
                os.path.join(self.gt_folder, f) 
                for f in os.listdir(self.gt_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
        else:
            for subset in os.listdir(self.gt_folder):
                subset_path = os.path.join(self.gt_folder, subset)
                if os.path.isdir(subset_path):
                    image_paths.extend([
                        os.path.join(subset_path, img) 
                        for img in os.listdir(subset_path) 
                        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ])
        
        return sorted(image_paths)  

    def _init_augmentator(self, augmentator):
        if augmentator is None:
            augmentators = parse_degradations(self.opt['augment'])
            return augmentators[0] if len(augmentators) == 1 else AugmentatorHub(augmentators)
        return augmentator

    def __getitem__(self, index):
        self._init_file_client()
        
        gt_path = self.paths[index]
        img_gt = self._load_image(gt_path)
        
        if self.opt['is_train']:
            img_gt = self._train_preprocessing(img_gt)
        else:
            img_gt = self._test_preprocessing(img_gt)
        
        img_lq = self.augmentator(img_gt)
        
        processed = self._process_images(img_gt, img_lq)
        
        return_dict = {
            'lq': processed['lq'],
            'gt': processed['gt'],
            'lq_path': gt_path,
            'gt_path': gt_path
        }
            
        return return_dict

    def _load_image(self, path):
        img_bytes = self.file_client.get(path)
        return imfrombytes(img_bytes, float32=True)

    def _train_preprocessing(self, img):
        img = augment(img, hflip=self.opt.get('use_hflip', False), rotation=False)
        return random_crop(img, self.opt['gt_size'])

    def _test_preprocessing(self, img):
        if self.opt.get('test_crop', False):
            return center_crop(img, self.opt['gt_size'])
        return img

    def __len__(self):
        return len(self.paths)
