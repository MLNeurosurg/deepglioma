import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


################ PLOTTING FUNCTION ##################
def plot_embedding(model,
                   dataset,
                   model_type='glove',
                   tsne=False,
                   perplexity=4):

    if model_type == 'glove':
        emb_i = model.wi.weight.cpu().data.numpy()
        emb_j = model.wj.weight.cpu().data.numpy()
        emb = (emb_i +
               emb_j) / 2  # emb_i and emb_j are asymptotically equivalent
    else:
        emb = model.embeddings_input.weight.data.cpu().data.numpy()

    if tsne:
        tsne = TSNE(metric='cosine', random_state=123, perplexity=perplexity)
        emb = tsne.fit_transform(emb)

    print(emb.shape)
    embedding_dict = {}
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(len(dataset._id2word)):
        embedding_dict[dataset._id2word[idx]] = (emb[idx, 0], emb[idx, 1])
        if 'wt' in dataset._id2word[idx]:
            continue
        else:
            plt.scatter(*emb[idx, :], color='steelblue')
            plt.annotate(dataset._id2word[idx], (emb[idx, 0], emb[idx, 1]),
                         alpha=0.7)
    plt.show()
    return embedding_dict


def plot_from_tran_model(model, labels, tsne=True, perplexity=4):

    emb = model.label_lt.weight.cpu().detach().numpy()
    if tsne:
        tsne = TSNE(metric='cosine', random_state=123, perplexity=perplexity)
        emb = tsne.fit_transform(emb)

    print(emb.shape)
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(len(labels)):
        plt.scatter(*emb[idx, :], color='steelblue')
        plt.annotate(labels[idx], (emb[idx, 0], emb[idx, 1]), alpha=0.7)
    plt.show()


###########################################


def weight_func(x, x_max, alpha):
    wx = (x / x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx


def rename_function(series, mut):
    for i, val in enumerate(series):
        if val == 0.0:
            series[i] = str(mut) + "wt"
        elif val == 1.0:
            series[i] = str(mut) + "mut"
        else:
            series[i] = "UNK"
    return series


def binary_to_string(dataframe):
    for column in dataframe.columns[1:]:
        dataframe[column] = rename_function(dataframe[column], column)
    return dataframe


def convert_df_to_dict(dataframe):

    index_dict = defaultdict(list)
    indices = dataframe["Sample ID"]
    dataframe.drop("Sample ID", inplace=True, axis=1)

    # look through the indices and columns
    for i, sample_id in enumerate(indices):
        for gene in dataframe.columns:
            index_dict[sample_id].append(dataframe.loc[i, gene])

    for sample, gene_list in index_dict.items():
        if len(gene_list) == 1:
            index_dict.pop(sample)

        if 'UNK' in gene_list:
            index_dict[sample] = [x for x in gene_list if x != 'UNK']

    return index_dict


def convert_to_strings(example_dict):
    string_list = []
    for _, gene_list in example_dict.items():
        string_list.append(' '.join(gene_list))
    return string_list
