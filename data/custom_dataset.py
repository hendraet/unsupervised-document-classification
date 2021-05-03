"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output


""" 
    DualNeighborsDataset
    Returns an image with one of its k nearest neighbors and one of its k furthest neighbors.
"""


class DualNeighborsDataset(Dataset):
    def __init__(self, dataset, knn_indices, kfn_indices, num_neighbors=None):
        super(DualNeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.knn_indices = knn_indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.kfn_indices = kfn_indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.knn_indices = self.knn_indices[:, :num_neighbors + 1]
            self.kfn_indices = self.kfn_indices[:, :int(len(dataset) * 0.75)]
        assert (self.knn_indices.shape[0] == self.kfn_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        kn_neighbor_index = np.random.choice(self.knn_indices[index], 1)[0]
        kn_neighbor = self.dataset.__getitem__(kn_neighbor_index)
        kf_neighbor_index = np.random.choice(self.kfn_indices[index], 1)[0]
        # kf_neighbor_index = np.random.randint(0, len(self.dataset))
        kf_neighbor = self.dataset.__getitem__(kf_neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        kn_neighbor['image'] = self.neighbor_transform(kn_neighbor['image'])
        kf_neighbor['image'] = self.neighbor_transform(kf_neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = kn_neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.knn_indices[index])
        output['furthest_neighbor'] = kf_neighbor['image']
        output['possible_furthest_neighbors'] = torch.from_numpy(self.kfn_indices[index])
        output['target'] = anchor['target']

        return output
