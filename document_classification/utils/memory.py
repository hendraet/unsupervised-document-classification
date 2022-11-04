"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C),
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy

        else:
            return indices

    def mine_negatives(self, k, calculate_accuracy=True):
        # mine k negatives for every sample
        features = self.features.cpu().numpy()

        sample_indices = np.random.randint(0, features.shape[0], (features.shape[0], 2))
        distances = np.linalg.norm(features[sample_indices[:, 0]] - features[sample_indices[:, 1]], axis=1)
        med = np.median(distances)

        negative_indices = np.zeros((features.shape[0], k), dtype=np.int32)
        negative_okay = np.zeros(negative_indices.shape, dtype=bool)
        step = 1000

        for iter in range(20):
            print(f'Starting iteration {iter}')
            negative_indices[~negative_okay] = np.random.randint(0, features.shape[0], negative_indices[~negative_okay].shape)

            for i in range(0, features.shape[0], step):
                print(f'Step {i} of iteration {iter}')
                f1 = self.features[i:i+step, np.newaxis, :].cuda()
                f2 = self.features[negative_indices[i:i+step, :], :].cuda()
                negative_okay[i:i+step, :] = torch.linalg.norm(f1 - f2, dim=2).cpu().numpy() > med

        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, negative_indices[:, ], axis=0)
            anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
            accuracy = np.mean(neighbor_targets != anchor_targets)
            return negative_indices, accuracy

        else:
            return negative_indices

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)

        assert(b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
