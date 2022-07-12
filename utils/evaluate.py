import numpy as np
import logging
from collections import OrderedDict
import torch
import math
from pdb import set_trace as stop
import os
from models.utils import custom_replace
from utils.metrics import *
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


def label_missing_vals(labels, preds, metric):
    label_vals = []
    for label_idx in range(labels.shape[1]):
        # 2 is the placeholder for missing values
        bool_mask = (labels[:, label_idx] != 2).flatten()
        try:
            val = metric(labels[bool_mask, label_idx], preds[bool_mask,
                                                             label_idx])
        except ValueError:
            print('Some labels raised errors.')
            continue
        label_vals.append(val)
    mean_val = np.array(label_vals).mean()
    return mean_val


def compute_metrics(all_predictions,
                    all_targets,
                    all_masks=None,
                    all_missing_masks=None,
                    loss=0,
                    loss_unk=0,
                    elapsed=0,
                    known_labels=0,
                    missing_values=False,
                    return_predictions=False,
                    verbose=True):

    # handle if predictions are stored as a dict
    if isinstance(all_predictions, dict) and isinstance(all_targets, dict):
        preds_store = torch.zeros(
            size=(len(all_predictions),
                  all_predictions[list(all_predictions.keys())[0]].shape[0]))
        targets_store = torch.zeros(
            size=(len(all_predictions),
                  all_predictions[list(all_predictions.keys())[0]].shape[0]))
        for i, (id, pred) in enumerate(all_predictions.items()):
            preds_store[i, :] = torch.tensor(pred)
            target = all_targets[id]
            targets_store[i, :] = torch.tensor(target)

        all_predictions = preds_store
        all_targets = targets_store

    # online batch statistics
    if verbose:
        preds = torch.ge(torch.sigmoid(all_predictions), 0.5).float()
        correct = (preds == all_targets).sum().cpu().numpy()
        total = (all_targets.size(0) * all_targets.size(1))
        print("Iteration: >>>> epoch accuracy: " + str(correct / total))

    # get unknown label masks
    if all_masks is not None:
        unknown_label_mask = custom_replace(all_masks, 1, 0, 0)

    # Compute mean average precision, corrected for label masking and missing values
    if missing_values:
        meanAP = label_missing_vals(all_targets, all_predictions,
                                    metrics.average_precision_score)
        meanAUC = label_missing_vals(all_targets, all_predictions,
                                     metrics.roc_auc_score)
    else:
        if known_labels > 0:
            meanAP = custom_mean_avg_precision(all_targets, all_predictions,
                                               unknown_label_mask)
            meanAUC = custom_mean_auc(all_targets, all_predictions,
                                      unknown_label_mask)
        else:
            meanAP = metrics.average_precision_score(all_targets,
                                                     all_predictions,
                                                     average='macro')
            meanAUC = metrics.roc_auc_score(all_targets,
                                            all_predictions,
                                            average='macro')

    optimal_threshold = 0.5
    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()
    if all_missing_masks is not None:
        all_missing_masks = all_missing_masks.numpy()

    # threshold the predictions
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1

    # compute F1 scores
    if missing_values:
        CF1 = f1_score(all_targets,
                       all_predictions_thresh,
                       all_missing_masks,
                       average='macro')
        OF1 = f1_score(all_targets,
                       all_predictions_thresh,
                       all_missing_masks,
                       average='micro')

    else:
        CP = metrics.precision_score(all_targets,
                                     all_predictions_thresh,
                                     average='macro')
        CR = metrics.recall_score(all_targets,
                                  all_predictions_thresh,
                                  average='macro')
        CF1 = (2 * CP * CR) / (CP + CR)
        OP = metrics.precision_score(all_targets,
                                     all_predictions_thresh,
                                     average='micro')
        OR = metrics.recall_score(all_targets,
                                  all_predictions_thresh,
                                  average='micro')
        OF1 = (2 * OP * OR) / (OP + OR)

    # compute example-based metrics
    acc_ = list(
        subset_accuracy(all_targets,
                        all_predictions_thresh,
                        all_missing_masks,
                        axis=1,
                        per_sample=True))
    hl_ = list(
        hamming_loss(all_targets,
                     all_predictions_thresh,
                     all_missing_masks,
                     axis=1,
                     per_sample=True))
    exf1_ = list(
        example_f1_score(all_targets,
                         all_predictions_thresh,
                         all_missing_masks,
                         axis=1,
                         per_sample=True))
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    eval_ret = OrderedDict([('Subset accuracy', acc),
                            ('Hamming accuracy', 1 - hl),
                            ('Example-based F1', exf1),
                            ('Label-based Micro F1', OF1),
                            ('Label-based Macro F1', CF1)])

    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('loss:  {:0.3f}'.format(loss))
        print('lossu: {:0.3f}'.format(loss_unk))
        print('----')
        print('mAP:   {:0.1f}'.format(meanAP * 100))
        print('mAUC:   {:0.1f}'.format(meanAUC * 100))
        print('----')
        print('SubsetAcc:   {:0.1f}'.format(ACC * 100))
        print('HammingAcc:   {:0.1f}'.format(HA * 100))
        print('EB-F1:   {:0.1f}'.format(ebF1 * 100))
        print('CF1:   {:0.1f}'.format(CF1 * 100))
        print('OF1:   {:0.1f}'.format(OF1 * 100))

    # store metrics
    metrics_dict = {}
    metrics_dict['mAP'] = meanAP
    metrics_dict['mAUC'] = meanAUC
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1

    if return_predictions:
        return metrics_dict, all_predictions, all_targets
    else:
        return metrics_dict


def print_results(labels, correct_dict, binary_pred_dict, label_dict):
    
    # print and log aggregation statistics
    label_count = np.zeros(len(labels))
    label_correct = np.zeros(len(labels))
    case_correct = {}
    for key, val in correct_dict.items():
        mask = label_dict[key] == 2
        label_count += ~mask
        # mask for missing values
        masked_binary_pred = np.ma.masked_array(binary_pred_dict[key],
                                                mask).astype(np.int64)
        masked_label = np.ma.masked_array(label_dict[key],
                                          mask).astype(np.int64)
        # find correct values
        agg_correct = (masked_binary_pred == masked_label).astype(int)
        label_correct += agg_correct.filled(
            0)  # fill the masked values with 0 for summation
        case_correct[key] = np.all(agg_correct)
        print(f'{key} >>> {val} >>> all correct {np.all(agg_correct)}')

    label_correct /= label_count
    print(f"Label-level accuracy >>> {label_correct}")
    total_correct = sum(label_correct) / len(labels)
    print(f"Total accuracy >>> {total_correct}")
    return total_correct
