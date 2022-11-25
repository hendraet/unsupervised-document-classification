"""
Authors: Wouter Van Gansbeke
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os

import numpy as np
import torch
from termcolor import colored

from document_classification.utils.common_config import get_model, get_train_dataset, \
    get_val_dataset, \
    get_val_dataloader, \
    get_val_transformations \
 \
    from document_classification.utils.config import create_config
from document_classification.utils.memory import MemoryBank
from document_classification.utils.utils import fill_memory_bank

# Parser
parser = argparse.ArgumentParser(description='Eval_nn')
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
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Checkpoint
    assert os.path.exists(p['pretext_checkpoint'])
    print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()
    
    # Save model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    knn_indices, knn_acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * knn_acc))
    np.save(p['topk_neighbors_train_path'], knn_indices)

    if p['compute_negatives']:
        topk = 200
        kfn_indices, kfn_acc = memory_bank_base.mine_negatives(topk)
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

