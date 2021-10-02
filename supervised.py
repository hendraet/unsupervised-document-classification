"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
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
from utils.evaluate_utils import get_predictions, supervised_evaluate
from utils.train_utils import supervised_train

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
    train_dataset = get_train_dataset(p, train_transformations,
                                      split='train', to_neighbors_dataset=False)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=False)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # Tensorboard writer
    writer = SummaryWriter(log_dir=p['scan_tb_dir'])

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p)
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

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
    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_acc = 0

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
        supervised_train(train_dataloader, model, criterion, optimizer, epoch, writer)

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on SCAN loss ...')
        supervised_stats = supervised_evaluate(predictions, writer, epoch)
        print(supervised_stats)
        accuracy = supervised_stats['accuracy']

        if accuracy > best_acc:
            print('New highest accuracy on validation set: %.4f -> %.4f' % (best_acc, accuracy))
            best_acc = accuracy
            torch.save({'model': model.module.state_dict()}, p['scan_model'])

        else:
            print('No new highest accuracy on validation set: %.4f -> %.4f' % (best_acc, accuracy))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_acc': best_acc},
                   p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions, features, thumbnails = get_predictions(p, val_dataloader, model,
                                                        return_features=True, return_thumbnails=True)
    writer.add_embedding(features, predictions[0]['targets'], thumbnails, p['epochs'], p['scan_tb_dir'])
    supervised_stats = supervised_evaluate(predictions)
    print(supervised_stats)


if __name__ == "__main__":
    main()
