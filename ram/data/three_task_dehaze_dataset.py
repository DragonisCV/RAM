import os
from ram.data.base_dataset import BaseDataset
from ram.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class DehazeOTSBETADataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None):
        super(DehazeOTSBETADataset, self).__init__(opt)
        self.gt_folder = gt_path or opt['dataroot_gt']
        self.lq_folder = lq_path or opt['dataroot_lq']
        self.augmentator = augmentator

        self.paths = []
        for root, _, files in os.walk(self.lq_folder):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.png')):
                    self.paths.append(os.path.join(root, fname))

        if self.opt.get('limit_ratio'):
            self._limit_dataset()

    def __getitem__(self, index):
        self._init_file_client()

        scale = self.opt.get('scale', 1)
        lq_path = self.paths[index]
        gt_path = self._get_gt_path(lq_path)

        img_gt = self._load_image(gt_path, 'gt')
        img_lq = self._load_image(lq_path, 'lq')

        if self.opt['phase'] == 'train':
            img_gt, img_lq = self._train_augmentation(img_gt, img_lq, scale, gt_path)
        else:
            img_gt, img_lq = self._test_processing(img_gt, img_lq, scale)

        if self.augmentator:
            img_lq = self.augmentator(img_lq)

        processed = self._process_images(img_gt, img_lq)

        return {
            'lq': processed['lq'],
            'gt': processed['gt'],
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def _get_gt_path(self, lq_path):
        lq_filename = os.path.basename(lq_path)
        base_name = lq_filename.split('_')[0] + '.jpg'
        return os.path.join(self.gt_folder, base_name)
