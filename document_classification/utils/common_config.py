"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms

from document_classification.data.augment import Augment, Cutout
from document_classification.utils.collate import collate_custom
from document_classification.utils.mypath import MyPath


def get_criterion(p):
    if p['criterion'] == 'simclr':
        from document_classification.losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'scan':
        from document_classification.losses.losses import SCANLoss
        criterion = SCANLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'mcl':
        from document_classification.losses.losses import MCLLoss
        criterion = MCLLoss()

    elif p['criterion'] == 'confidence-cross-entropy':
        from document_classification.losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    elif p['criterion'] == 'binary-cross-entropy':
        criterion = torch.nn.BCELoss()

    elif p['criterion'] == 'categorical-cross-entropy':
        criterion = torch.nn.NLLLoss()

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None, load_simpred=False):
    # Get backbone
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-20']:
            from document_classification.models.resnet_cifar import resnet18
            backbone = resnet18()

        elif p['train_db_name'] == 'stl-10':
            from document_classification.models.resnet_stl import resnet18
            backbone = resnet18()

        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from document_classification.models.resnet import resnet50
            backbone = resnet50()

        elif p['train_db_name'] in ['impact_kb', 'impact_full_balanced', 'impact_full_imbalanced',
                                    'hdi_balanced', 'hdi_imbalanced', 'tobacco3482', 'rvl-cdip', 'wpi_demo']:
            from document_classification.models.resnet import resnet50
            backbone = resnet50()

        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet34':
        if p['train_db_name'] in ['impact_kb', 'impact_full_balanced', 'impact_full_imbalanced']:
            from document_classification.models.resnet import resnet34
            backbone = resnet34()

        else:
            raise NotImplementedError

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from document_classification.models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] == 'simpred' or load_simpred:
        from document_classification.models.models import SimpredModel
        model = SimpredModel(backbone, p['hidden_dim'])

    elif p['setup'] in ['scan', 'selflabel']:
        from document_classification.models.models import ClusteringModel
        if p['setup'] == 'selflabel':
            assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if load_simpred:
            missing = model.load_state_dict(state['model'], strict=False)
            assert (len(missing.missing_keys) == len(missing.unexpected_keys) == 0)

        elif p['setup'] in ['scan', 'simpred']: # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert(set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias', 
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                or set(missing[1]) == {
                'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel': # Weights are supposed to be transfered from scan 
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.0.weight' %(state['head'])]
            best_head_bias = model_state['cluster_head.%d.0.bias' %(state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.0.weight'] = best_head_weight
            model_state['cluster_head.0.0.bias'] = best_head_bias
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_train_dataset(p, transform, to_augmented_dataset=False, to_neighbors_dataset=False,
                      to_similarity_dataset=False, split=None, use_negatives=False, use_simpred=False):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from document_classification.data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'cifar-20':
        from document_classification.data.cifar import CIFAR20
        dataset = CIFAR20(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'stl-10':
        from document_classification.data.stl import STL10
        dataset = STL10(split=split, transform=transform, download=True)

    elif p['train_db_name'] in ['impact_kb', 'impact_full_balanced', 'impact_full_imbalanced',
                                'hdi_balanced', 'hdi_imbalanced', 'tobacco3482', 'rvl-cdip', 'wpi_demo']:
        from document_classification.data.imagefolderwrapper import ImageFolderWrapper
        root = MyPath.db_root_dir(p['train_db_name'])
        dataset = ImageFolderWrapper(root, split="train", transform=transform)

    elif p['train_db_name'] == 'imagenet':
        from document_classification.data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)

    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from document_classification.data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns an image and an augmentation of that image.
        from document_classification.data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset:  # Dataset returns an image and one of its nearest neighbors.
        from document_classification.data.custom_dataset import NeighborsDataset
        knn_indices = np.load(p['topk_neighbors_train_path'])

        if use_negatives:
            kfn_indices = np.load(p['topk_furthest_train_path'])
            num_negatives = p['num_negatives']
        else:
            kfn_indices = None
            num_negatives = None

        dataset = NeighborsDataset(dataset, knn_indices, kfn_indices, use_simpred, p['num_neighbors'], num_negatives)
    elif to_similarity_dataset:  # Dataset returns an image and another random image.
        from document_classification.data.custom_dataset import SimilarityDataset
        dataset = SimilarityDataset(dataset)
    
    return dataset

def get_val_dataset(p, transform=None, to_neighbors_dataset=False, to_similarity_dataset=False,
                    use_negatives=False, use_simpred=False):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from document_classification.data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)
    
    elif p['val_db_name'] == 'cifar-20':
        from document_classification.data.cifar import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from document_classification.data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)

    elif p['train_db_name'] in ['impact_kb', 'impact_full_balanced', 'impact_full_imbalanced',
                                'hdi_balanced', 'hdi_imbalanced', 'tobacco3482', 'rvl-cdip', 'wpi_demo']:
        from document_classification.data.imagefolderwrapper import ImageFolderWrapper
        root = MyPath.db_root_dir(p['train_db_name'])
        dataset = ImageFolderWrapper(root, split="test", transform=transform)
    
    elif p['val_db_name'] == 'imagenet':
        from document_classification.data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)
    
    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from document_classification.data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)
    
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from document_classification.data.custom_dataset import NeighborsDataset
        knn_indices = np.load(p['topk_neighbors_val_path'])

        if use_negatives:
            kfn_indices = np.load(p['topk_furthest_val_path'])
        else:
            kfn_indices = None

        dataset = NeighborsDataset(dataset, knn_indices, kfn_indices, use_simpred, 5, 5) # Only use 5
    elif to_similarity_dataset:  # Dataset returns an image and another random image.
        from document_classification.data.custom_dataset import SimilarityDataset
        dataset = SimilarityDataset(dataset)

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=False, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=False, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])
    
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(), 
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr