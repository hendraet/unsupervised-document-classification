"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, \
    get_val_dataset, get_val_dataloader, get_val_transformations
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--tb_run', help='Tensorboard log directory')
args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.tb_run)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, transforms)
    val_dataset = get_val_dataset(p, transforms)
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    memory_bank_train = MemoryBank(len(train_dataset), 2048, p['num_classes'], p['temperature'])
    memory_bank_train.cuda()
    memory_bank_val = MemoryBank(len(val_dataset), 2048, p['num_classes'], p['temperature'])
    memory_bank_val.cuda()

    # Load the official MoCoV2 checkpoint
    print(colored('Downloading moco v2 checkpoint', 'blue'))
    moco_state = torch.load(p['pretrained'], map_location='cpu')

    # Transfer moco weights
    print(colored('Transfer MoCo weights to model', 'blue'))
    new_state_dict = {}
    state_dict = moco_state['state_dict']
    for k in list(state_dict.keys()):
        # Copy backbone weights
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            new_k = 'module.backbone.' + k[len('module.encoder_q.'):]
            new_state_dict[new_k] = state_dict[k]

        # Copy mlp weights
        elif k.startswith('module.encoder_q.fc'):
            new_k = 'module.contrastive_head.' + k[len('module.encoder_q.fc.'):]
            new_state_dict[new_k] = state_dict[k]

        else:
            pass  # just silently discard unexpected keys
            # raise ValueError('Unexpected key {}'.format(k))

    model.load_state_dict(new_state_dict)

    # Save final model
    print(colored('Save pretext model', 'blue'))
    torch.save(model.module.state_dict(), p['pretext_model'])
    model.module.contrastive_head = torch.nn.Identity()  # In this case, we mine the neighbors before the MLP.

    # Mine the topk nearest neighbors (Train)
    # These will be used for training with the SCAN-Loss.
    topk = 50
    print(colored('Mine the nearest neighbors (Train)(Top-%d)' % (topk), 'blue'))
    transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, transforms)
    fill_memory_bank(train_dataloader, model, memory_bank_train)
    knn_indices, knn_acc = memory_bank_train.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * knn_acc))
    np.save(p['topk_neighbors_train_path'], knn_indices)

    if p['compute_negatives']:
        topk = 350
        kfn_indices, kfn_acc = memory_bank_train.mine_negatives(topk)
        print('Accuracy of top-%d furthest neighbors on train set is %.2f' % (topk, 100 * kfn_acc))
        np.save(p['topk_furthest_train_path'], kfn_indices)

    # Mine the topk nearest neighbors (Validation)
    # These will be used for validation.
    topk = 5
    print(colored('Mine the nearest neighbors (Val)(Top-%d)' % (topk), 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    print('Mine the neighbors')
    knn_indices, knn_acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * knn_acc))
    np.save(p['topk_neighbors_val_path'], knn_indices)

    if p['compute_negatives']:
        kfn_indices, kfn_acc = memory_bank_val.mine_negatives(topk)
        print('Accuracy of top-%d furthest neighbors on val set is %.2f' % (topk, 100 * kfn_acc))
        np.save(p['topk_furthest_val_path'], kfn_indices)


if __name__ == '__main__':
    main()
