"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        heads = [nn.Sequential(nn.Linear(self.backbone_dim, nclusters), nn.Softmax(dim=-1)) for _ in range(self.nheads)]
        self.cluster_head = nn.ModuleList(heads)

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out


class SimpredModel(nn.Module):
    def __init__(self, backbone, hidden_dim=512):
        super(SimpredModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = nn.Sequential(nn.Linear(self.backbone_dim * 2, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, 1),
                                  nn.Sigmoid())

    def forward(self, x1, x2, forward_pass='default'):
        if forward_pass == 'default':
            features1 = self.backbone(x1)
            features2 = self.backbone(x2)

            concatenated = torch.cat((features1, features2), dim=1)

            out = [self.head(concatenated)]

        elif forward_pass == 'backbone':
            features1 = self.backbone(x1)
            features2 = self.backbone(x2)
            out = features1, features2

        elif forward_pass == 'head':
            concatenated = torch.cat((x1, x2), dim=1)
            out = [self.head(concatenated)]

        elif forward_pass == 'return_all':
            features1 = self.backbone(x1)
            features2 = self.backbone(x2)

            concatenated = torch.cat((features1, features2), dim=1)

            out = {'features': (features1, features2), 'output': [self.head(concatenated)]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
