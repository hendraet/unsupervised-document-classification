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
    def __init__(self, dataset, knn_indices, kfn_indices, num_neighbors=None):
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
        self.knn_indices = knn_indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.kfn_indices = kfn_indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.knn_indices = self.knn_indices[:, :num_neighbors + 1]
            self.kfn_indices = self.kfn_indices[:, :num_neighbors + 1]
        assert (self.knn_indices.shape[0] == len(self.dataset))
        assert (self.kfn_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return self.knn_indices.shape[0] * self.knn_indices.shape[1] + \
               self.kfn_indices.shape[0] * self.kfn_indices.shape[1]

    def _map_index(self, index):
        if index < self.knn_indices.shape[0] * self.knn_indices.shape[1]:
            label = 1.0
            anchor = index // self.knn_indices.shape[1]
            neighbor_index = index - anchor * self.knn_indices.shape[1]
            neighbor = self.knn_indices[anchor, neighbor_index]
        else:
            label = 0.0

            index_new = index - self.knn_indices.shape[0] * self.knn_indices.shape[1]
            anchor = index_new // self.kfn_indices.shape[1]
            neighbor_index = index_new - anchor * self.kfn_indices.shape[1]
            neighbor = self.kfn_indices[anchor, neighbor_index]

        return anchor, neighbor, label

    def __getitem__(self, index):
        output = {}

        anchor_index, neighbor_index, label = self._map_index(index)

        anchor = self.dataset.__getitem__(anchor_index)
        anchor['image'] = self.anchor_transform(anchor['image'])
        
        # neighbor_index = np.random.choice(self.knn_indices[index])
        neighbor = self.dataset.__getitem__(neighbor_index)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        # label = 1.0 if anchor['target'] == neighbor['target'] else 0.0

        # furthest_neighbor_index = np.random.choice(self.kfn_indices[index])
        # furthest_neighbor = self.dataset.__getitem__(furthest_neighbor_index)['image']
        # furthest_neighbor = self.neighbor_transform(furthest_neighbor)

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['label'] = label
        # output['furthest_neighbor'] = furthest_neighbor
        output['possible_neighbors'] = torch.from_numpy(self.knn_indices[anchor_index])
        # output['possible_furthest_neighbors'] = torch.from_numpy(self.kfn_indices[index])
        output['target'] = anchor['target']
        
        return output

""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class SimilarityDataset(Dataset):
    def __init__(self, dataset):
        super(SimilarityDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}

        anchor = self.dataset.__getitem__(index)
        anchor['image'] = self.anchor_transform(anchor['image'])

        neighbor_index = np.random.randint(len(self))
        neighbor = self.dataset.__getitem__(neighbor_index)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        label = anchor['target'] == neighbor['target']

        output['image'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['target'] = label

        return output
