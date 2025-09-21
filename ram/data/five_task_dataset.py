from torch.utils import data as data
from ram.data.base.dehaze_dataset import DehazeOTSBETADataset # Share the same dataset with 3task
from ram.data.base.derain_dataset import Rain100LTrainDataset # Share the same dataset with 3task
from ram.data.base.low_cost_dataset import LowCostNoiseDataset # Share the same dataset with 3task
from ram.data.base.motionblur_dataset import GoProDataset
from ram.data.base.lowlight_dataset import LOLv1Dataset
from ram.utils.registry import DATASET_REGISTRY
from ram.data.utils.online_util import parse_degradations
import random
import torch

@DATASET_REGISTRY.register()
class fiveTaskDataset(data.Dataset):
    def __init__(self, opt):
        super(fiveTaskDataset, self).__init__()
        self.opt = opt

        self.ots_dataset = DehazeOTSBETADataset(opt, lq_path=opt['ots_lq_path'], gt_path=opt['ots_gt_path'],enlarge_ratio=opt['haze_enlarge_ratio'])
        self.rain100L_dataset = Rain100LTrainDataset(opt, dataroot=opt['rain_path'],enlarge_ratio=opt['rain_enlarge_ratio'])

        augmentators1 = parse_degradations(opt['augment1'])
        noise_dataset1 = [LowCostNoiseDataset(opt, dataroot=opt['bsd_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators1]
        noise_dataset2 = [LowCostNoiseDataset(opt, dataroot=opt['wed_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators1]
        augmentators2 = parse_degradations(opt['augment2'])
        noise_dataset3 = [LowCostNoiseDataset(opt, dataroot=opt['bsd_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators2]
        noise_dataset4 = [LowCostNoiseDataset(opt, dataroot=opt['wed_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators2]
        augmentators3 = parse_degradations(opt['augment3'])
        noise_dataset5 = [LowCostNoiseDataset(opt, dataroot=opt['bsd_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators3]
        noise_dataset6 = [LowCostNoiseDataset(opt, dataroot=opt['wed_path'], augmentator=augmentator,enlarge_ratio=opt['noise_enlarge_ratio']) for augmentator in augmentators3]

        self.deblur_dataset = GoProDataset(opt,dataroot=opt['gopro_path'],enlarge_ratio=opt['motionblur_enlarge_ratio'])
        self.lolv1_dataset = LOLv1Dataset(opt,lq_path=opt['lol_v1_lq_path'],gt_path=opt['lol_v1_gt_path'],enlarge_ratio=opt['lowlight_enlarge_ratio'])
        self.datasets = [self.ots_dataset, self.rain100L_dataset, self.deblur_dataset,self.lolv1_dataset] + \
                        noise_dataset1 + noise_dataset2 + noise_dataset3 + noise_dataset4 + noise_dataset5 + noise_dataset6

        self.sample_ids = []
        for i, dataset in enumerate(self.datasets):
            length = len(dataset)
            for idx in range(length):
                self.sample_ids.append({'dataset_idx': i, 'sample_idx': idx})

        random.shuffle(self.sample_ids)

    def __getitem__(self, index):
        sample_info = self.sample_ids[index]
        dataset_idx = sample_info['dataset_idx']
        sample_idx = sample_info['sample_idx']

        sample = self.datasets[dataset_idx][sample_idx]
        sample['degradation_type'] = torch.tensor(dataset_idx, dtype=torch.long)
        return sample

    def __len__(self):
        return len(self.sample_ids)
