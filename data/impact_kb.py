import os
import torchvision

from utils.mypath import MyPath


class IMPACT_KB(torchvision.datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('impact_kb'), split="train", transform=None, target_transform=None):
        super().__init__(os.path.join(root, split), transform, target_transform)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        sample = dict()
        sample['image'] = image
        sample['target'] = target

        return sample
