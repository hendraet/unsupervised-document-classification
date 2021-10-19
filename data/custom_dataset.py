"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from data.imagefolderwrapper import ImageFolderWrapper
from data.stl import STL10

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
    def __init__(self, dataset, knn_indices, negative_indices=None, use_simpred=False, num_neighbors=None, num_negatives=None):
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
        self.negative_indices = negative_indices  # Negative indices (np.array  [len(dataset) x k])

        self.use_simpred = use_simpred
        if num_neighbors is not None:
            self.knn_indices = self.knn_indices[:, :num_neighbors + 1]
        assert (self.knn_indices.shape[0] == len(self.dataset))

        if self.negative_indices is not None:
            if num_neighbors is not None:
                self.negative_indices = self.negative_indices[:, :num_neighbors + 1]
            self.positive_ratio = self.knn_indices.shape[1] / (self.knn_indices.shape[1] + self.negative_indices.shape[1])
            assert (self.negative_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}

        if self.use_simpred:
            query_index = np.random.randint(len(self))
            label = 0.0  # Placeholder, simpred output will be used instead
        elif self.negative_indices is None:
            query_index = np.random.choice(self.knn_indices[index])
            label = 1.0
        else:
            rand = np.random.random_sample()
            # Decide whether to sample a positive or a negative
            if rand < self.positive_ratio:
                query_index = np.random.choice(self.knn_indices[index])
                label = 1.0
            else:
                query_index = np.random.choice(self.negative_indices[index])
                label = 0.0

        anchor = self.dataset.__getitem__(index)
        anchor['image'] = self.anchor_transform(anchor['image'])
        
        query = self.dataset.__getitem__(query_index)
        query['image'] = self.neighbor_transform(query['image'])

        output['anchor'] = anchor['image']
        output['query'] = query['image']
        output['label'] = label
        output['possible_neighbors'] = torch.from_numpy(self.knn_indices[index])
        output['target'] = anchor['target']
        
        return output

""" 
    SimilarityDataset
    Returns an image with one of its neighbors as well as their similarity.
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

        if isinstance(self.dataset, STL10):
            labels = self.dataset.labels
        elif isinstance(self.dataset, ImageFolderWrapper):
            labels = [sample[1] for sample in self.dataset.imgs]
            labels = np.array(labels)

        num_classes = np.unique(labels).shape[0]
        self.indices_by_class = [[] for _ in range(num_classes)]
        indices = np.arange(len(self))

        for i in range(num_classes):
            self.indices_by_class[i] = indices[labels == i]

        self.cls_distribution = np.array([cls_indices.shape[0] for cls_indices in self.indices_by_class])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}

        anchor = self.dataset.__getitem__(index)
        anchor['image'] = self.anchor_transform(anchor['image'])

        # Decide whether to return a positive or a negative
        positive = np.random.random_sample() > 0.5

        if positive:
            neighbor_index = np.random.choice(self.indices_by_class[anchor['target']])
            neighbor = self.dataset.__getitem__(neighbor_index)
            neighbor['image'] = self.neighbor_transform(neighbor['image'])
        else:
            dist = self.cls_distribution.copy()
            dist[anchor['target']] = 0
            dist = dist / dist.sum()
            negative_cls = np.random.choice(np.arange(len(self.indices_by_class)), p=dist)
            negative_index = np.random.choice(self.indices_by_class[negative_cls])
            neighbor = self.dataset.__getitem__(negative_index)
            neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['image'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['target'] = float(positive)

        return output
