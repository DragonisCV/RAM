from ram.data.utils.data_util import paired_paths_from_folder
from ram.utils.registry import DATASET_REGISTRY
from ram.data.base_dataset import BaseDataset

@DATASET_REGISTRY.register()
class PairedImageDataset(BaseDataset):
    def __init__(self, opt, lq_path=None, gt_path=None, augmentator=None):
        super(PairedImageDataset, self).__init__(opt)
        self.gt_folder = gt_path or opt['dataroot_gt']
        self.lq_folder = lq_path or opt['dataroot_lq']
        self.augmentator = augmentator
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], 
            ['lq', 'gt'],
            self.opt.get('filename_tmpl', '{}')
        )

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
    def __len__(self):
        return len(self.paths)
