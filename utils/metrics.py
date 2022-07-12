#!/usr/bin/env python3
"""Helper functions to compute all metrics needed"""
import os
import sys
import numpy as np
import logging
from typing import List, Dict
from collections import OrderedDict, defaultdict

from sklearn import metrics
from threading import Lock
import pandas as pd
import torch
import math


def aggregate_predictions(all_ids: List,
                          all_preds: torch.Tensor,
                          all_labels: torch.Tensor,
                          all_masks: torch.Tensor,
                          all_missing_masks: torch.Tensor,
                          threshold: float = 0.5,
                          prediction_level: str = 'patient') -> Dict:

    # zip results
    results = list(zip(all_ids, all_preds, all_labels, all_missing_masks))
    # iterate over predictions
    correct_patch_dict = dict()
    aggregation_dict = dict()
    binary_pred_dict = dict()
    label_dict = dict()
    mask_dict = dict()

    count_dict = defaultdict(int)
    for patch, pred, label, mask in results:
        if prediction_level is 'patient':
            key = patch.split('/')[1]
        elif prediction_level is 'series':
            key = patch.split('/')[1] + '/' + patch.split('/')[2]

        if key not in label_dict.keys():
            label_dict[key] = label.numpy()
            mask_dict[key] = mask.numpy()
            aggregation_dict[key] = 0
            correct_patch_dict[key] = 0

        # get predictions
        pred = torch.sigmoid(pred)
        binary_pred = torch.ge(pred, 0.5).numpy()
        correct = (binary_pred == label.numpy())

        # populate dictionaries
        aggregation_dict[key] += pred.numpy()
        correct_patch_dict[key] += correct
        count_dict[key] += 1

    # normalize the predictions
    for key, val in correct_patch_dict.items():
        correct_patch_dict[key] = val / count_dict[key]
        # handle missing values
        correct_patch_dict[key][mask_dict[key] == 0] = np.nan
    for key, val in aggregation_dict.items():
        pred = val / count_dict[key]
        aggregation_dict[key] = pred
        binary_pred_dict[key] = np.where(pred > threshold, 1, 0)

    return aggregation_dict, correct_patch_dict, binary_pred_dict, label_dict


def postprocess_features_tsne(all_ids: List,
                              all_preds: torch.Tensor,
                              all_labels: torch.Tensor,
                              all_features: torch.Tensor,
                              n_labels: int = 3):

    # zip results
    results = list(zip(all_ids, all_preds, all_labels, all_features))

    # iterate over predictions
    aggregation_dict = defaultdict(list)
    for patch, pred, label, features in results:
        patient = patch.split('/')[1]
        mosaic = f"{patch.split('/')[1]}/{patch.split('/')[2]}"

        pred = torch.sigmoid(pred)
        binary_pred = torch.ge(pred, 0.5).numpy()
        correct = (binary_pred == label.numpy())

        aggregation_dict['patch'].append(patch)
        aggregation_dict['patients'].append(patient)
        aggregation_dict['mosaic'].append(mosaic)
        aggregation_dict['pred'].append(pred.numpy())
        aggregation_dict['labels'].append(label.numpy())
        aggregation_dict['features'].append(features.numpy())
        aggregation_dict['correct'].append(correct)
        aggregation_dict[f'label_idhnoncodel'].append(
            (label[0] * 1 - label[1]).numpy())

        for i in range(n_labels):
            aggregation_dict[f'label_{i}'].append(label[i].numpy())

    return aggregation_dict


def downsample_tsne_dict(tsne_dict, downsample_factor=0.5):

    sample_interval = int(1 / downsample_factor)

    downsampled_dict = {}
    for key, val in tsne_dict.items():
        downsampled_dict[key] = val[::sample_interval]

    return downsampled_dict


def error_rate(true_targets, predictions):
    acc = metrics.accuracy_score(true_targets, predictions)
    error_rate = 1 - acc
    return error_rate


def subset_accuracy(true_targets,
                    predictions,
                    missing_mask=None,
                    per_sample=False,
                    axis=0):

    if missing_mask is not None:
        true_target = np.ma.masked_array(true_targets,
                                         mask=~missing_mask.astype(bool))
        predictions = np.ma.masked_array(predictions,
                                         mask=~missing_mask.astype(bool))
        result = np.all(true_targets == predictions, axis=axis)
    else:
        result = np.all(true_targets == predictions, axis=axis)

    if not per_sample:
        result = np.mean(result)
    return result


def hamming_loss(true_targets,
                 predictions,
                 missing_mask=None,
                 per_sample=False,
                 axis=0):

    if missing_mask is not None:
        true_target = np.ma.masked_array(true_targets,
                                         mask=~missing_mask.astype(bool))
        predictions = np.ma.masked_array(predictions,
                                         mask=~missing_mask.astype(bool))
        result = np.mean(np.logical_xor(true_targets, predictions), axis=axis)
    else:
        result = np.mean(np.logical_xor(true_targets, predictions), axis=axis)

    if not per_sample:
        result = np.mean(result)
    return result


def compute_tp_fp_fn(true_targets, predictions, missing_mask=None, axis=0):
    # axis: axis for instance
    if missing_mask is not None:
        true_target = np.ma.masked_array(true_targets,
                                         mask=~missing_mask.astype(bool))
        predictions = np.ma.masked_array(predictions,
                                         mask=~missing_mask.astype(bool))

    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions,
                axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions),
                axis=axis).astype('float32')
    return (tp, fp, fn)


def example_f1_score(true_targets,
                     predictions,
                     missing_mask=None,
                     per_sample=False,
                     axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets,
                                  predictions,
                                  missing_mask,
                                  axis=axis)

    numerator = 2 * tp
    denominator = (np.sum(true_targets, axis=axis).astype('float32') +
                   np.sum(predictions, axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]
    denominator = np.delete(denominator, zeros)
    numerator = np.delete(numerator, zeros)
    example_f1 = numerator / denominator

    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1


def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro.")

    if average == 'micro':
        f1 = (2 * np.sum(tp)) / ((2 * np.sum(tp)) + (np.sum(fp) + np.sum(fn)))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]

        # f1 = np.mean(safe_div(2*tp, (2*tp) + (fp + fn)))
        f1 = np.mean(np.divide(2 * tp, (2 * tp) + (fp + fn)))

    return f1


def f1_score(true_targets,
             predictions,
             missing_mask=None,
             average='micro',
             axis=0):
    """
		average: str
			'micro' or 'macro'
		axis: 0 or 1
			label axis
	"""
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets,
                                  predictions,
                                  missing_mask,
                                  axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def custom_mean_avg_precision(all_targets, all_predictions,
                              unknown_label_mask):
    APs = []
    for label_idx in range(all_targets.size(1)):
        all_targets_unk = torch.masked_select(
            all_targets[:, label_idx],
            unknown_label_mask[:, label_idx].type(torch.ByteTensor))
        all_predictions_unk = torch.masked_select(
            all_predictions[:, label_idx],
            unknown_label_mask[:, label_idx].type(torch.ByteTensor))
        if len(all_targets_unk) > 0 and all_targets_unk.sum().item() > 0:
            AP = metrics.average_precision_score(all_targets_unk,
                                                 all_predictions_unk,
                                                 average=None,
                                                 pos_label=1)
            APs.append(AP)
    meanAP = np.array(APs).mean()
    return meanAP


def custom_mean_auc(all_targets, all_predictions, unknown_label_mask):
    rocs = []
    for label_idx in range(all_targets.size(1)):
        all_targets_unk = torch.masked_select(
            all_targets[:, label_idx],
            unknown_label_mask[:, label_idx].type(torch.ByteTensor))
        all_predictions_unk = torch.masked_select(
            all_predictions[:, label_idx],
            unknown_label_mask[:, label_idx].type(torch.ByteTensor))
        if len(all_targets_unk) > 0 and all_targets_unk.sum().item() > 0:
            roc = metrics.roc_auc_score(all_targets_unk,
                                        all_predictions_unk,
                                        average=None)
            rocs.append(roc)
    meanAP = np.array(rocs).mean()
    return meanAP
