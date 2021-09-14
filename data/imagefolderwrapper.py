import os
import torchvision


class ImageFolderWrapper(torchvision.datasets.ImageFolder):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(os.path.join(root, split), transform, target_transform)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        sample = dict()
        sample['image'] = image
        sample['target'] = target

        return sample
