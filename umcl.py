"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Modified by Jona Otholt
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch

from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations, \
    get_train_dataset, get_train_dataloader, \
    get_val_dataset, get_val_dataloader, \
    get_optimizer, get_model, get_criterion, \
    adjust_learning_rate
from utils.evaluate_utils import get_predictions, hungarian_evaluate, umcl_evaluate
from utils.train_utils import umcl_train

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--tb_run', help='Tensorboard log directory')


def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp, args.tb_run)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, use_negatives=not p['use_simpred_model'],
                                      use_simpred=p['use_simpred_model'], split='train', to_neighbors_dataset=True)
    val_dataset = get_val_dataset(p, val_transformations, use_negatives=not p['use_simpred_model'],
                                  use_simpred=p['use_simpred_model'], to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # Tensorboard writer
    writer = SummaryWriter(log_dir=p['scan_tb_dir'])

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Simpred Model
    if p['use_simpred_model']:
        print(colored('Get simpred model', 'blue'))
        simpred_model = get_model(p, p['simpred_model'], load_simpred=True)
        print(simpred_model)
        simpred_model = torch.nn.DataParallel(simpred_model)
        simpred_model = simpred_model.cuda()
        for param in simpred_model.parameters():
            param.requires_grad = False
    else:
        print('Not using simpred model')
        simpred_model = None

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # Checkpoint
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_acc_head = checkpoint['best_acc_head']

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_acc = 0
        best_acc_head = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        umcl_train(train_dataloader, model, simpred_model, criterion, optimizer,
                   epoch, writer, p['update_cluster_head_only'])

        # Evaluate 
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on similarity accuracy')
        stats = umcl_evaluate(p, val_dataloader, model, simpred_model)
        print(stats)
        highest_acc_head = stats['highest_acc_head']
        highest_acc = stats['highest_acc']

        if highest_acc > best_acc:
            print('New highest accuracy on validation set: %.4f -> %.4f' % (best_acc, highest_acc))
            print('Highest accuracy head is %d' % highest_acc_head)
            best_acc = highest_acc
            best_acc_head = highest_acc_head
            torch.save({'model': model.module.state_dict(), 'head': best_acc_head}, p['scan_model'])

        else:
            print('No new highest accuracy on validation set: %.4f -> %.4f' % (best_acc, highest_acc))
            print('Highest accuracy head is %d' % highest_acc_head)

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(highest_acc_head, predictions,
                                              compute_confusion_matrix=False, tf_writer=writer, epoch=epoch)
        print(clustering_stats)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_acc': best_acc, 'best_acc_head': best_acc_head},
                   p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on similarity accuracy at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions, features, thumbnails = get_predictions(p, val_dataloader, model,
                                                        return_features=True, return_thumbnails=True)
    writer.add_embedding(features, predictions[0]['targets'], thumbnails, p['epochs'], p['scan_tb_dir'])
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions,
                                          class_names=val_dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png'))
    print(clustering_stats)


if __name__ == "__main__":
    main()
