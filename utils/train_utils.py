"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, writer, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbors'].cuda(non_blocking=True)
        neighbors = neighbors.reshape(-1, neighbors.shape[2], neighbors.shape[3], neighbors.shape[4])

        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

            # Loss for every head
        loss = []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            loss_ = criterion(anchors_output_subhead, neighbors_output_subhead)

            loss.append(loss_)

            num_classes = anchors_output_subhead.shape[1]
            writer.add_scalar('Train/Loss/Head-%d' % num_classes, loss_.item(), epoch * len(train_loader) + i)

        # Register the mean loss and backprop the total loss to cover all subheads
        losses.update(np.mean([v.item() for v in loss]))

        total_loss = torch.sum(torch.stack(loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, writer, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        num_classes = output_augmented.shape[1]
        writer.add_scalar('Train/Loss/Head-%d' % num_classes, loss.item(), epoch * len(train_loader) + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:  # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)
