"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset, DualNeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy, xentropy


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output) 

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    neighbor_probs = [[] for _ in range(p['num_heads'])]
    furthest_neighbor_probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    
    if isinstance(dataloader.dataset, (NeighborsDataset, DualNeighborsDataset)):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr+bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        targets.append(batch['target'])

        if include_neighbors:
            images = batch['neighbor'].cuda(non_blocking=True)
            output = model(images)
            for i, output_i in enumerate(output):
                neighbor_probs[i].append(F.softmax(output_i, dim=1))

            if isinstance(dataloader.dataset, DualNeighborsDataset):
                images = batch['furthest_neighbor'].cuda(non_blocking=True)
                output = model(images)
                for i, output_i in enumerate(output):
                    furthest_neighbor_probs[i].append(F.softmax(output_i, dim=1))

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbor_probs = [torch.cat(prob_, dim=0).cpu() for prob_ in neighbor_probs]
        if isinstance(dataloader.dataset, DualNeighborsDataset):
            furthest_neighbor_probs = [torch.cat(prob_, dim=0).cpu() for prob_ in furthest_neighbor_probs]
            # neighbors = torch.cat(neighbors, dim=0)
            # furthest_neighbors = torch.cat(furthest_neighbors, dim=0)
            out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'furthest_neighbors': fn_prob_, 'neighbors': n_prob_} for
                   pred_, prob_, n_prob_, fn_prob_ in zip(predictions, probs, neighbor_probs, furthest_neighbor_probs)]
        else:
            # neighbors = torch.cat(neighbors, dim=0)
            out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': n_prob_} for
                   pred_, prob_, n_prob_ in zip(predictions, probs, neighbor_probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def scan_evaluate(predictions, criterion):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbor_probs = head['neighbors']
        furthest_neighbor_probs = None

        if 'furthest_neighbors' in head:
            furthest_neighbor_probs = head['furthest_neighbors']

        total_loss, consistency_loss, entropy_loss = criterion.from_probabilities(probs, neighbor_probs, furthest_neighbor_probs)
        
        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None, compute_purity=True,
                       compute_confusion_matrix=True, confusion_matrix_file=None, tf_writer=None, epoch=0):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    _, preds_top5 = probs.topk(min(5, num_classes), 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    reordered_preds = reordered_preds.cpu().numpy()
    targets = targets.cpu().numpy()

    if tf_writer is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(targets, reordered_preds, average=None, zero_division=0)
        recall = recall_score(targets, reordered_preds, average=None, zero_division=0)
        f1 = f1_score(targets, reordered_preds, average=None, zero_division=0)

        tf_writer.add_scalar('Evaluate/ACC', acc, epoch)
        tf_writer.add_scalar('Evaluate/NMI', nmi, epoch)
        tf_writer.add_scalar('Evaluate/ARI', ari, epoch)

        for i in range(len(f1)):
            tf_writer.add_scalar(f'Evaluate/f1_{i}', f1[i], epoch)
            tf_writer.add_scalar(f'Evaluate/precision_{i}', precision[i], epoch)
            tf_writer.add_scalar(f'Evaluate/recall_{i}', recall[i], epoch)

        # if epoch % cfg.embedding_freq == 0:
        #     tf_writer.add_embedding(intermediates, labels, images, epoch, cfg.session)

    # Visualize confusion matrix with matplotlib
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds, targets, class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res
