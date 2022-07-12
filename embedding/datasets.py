"""PyTorch datasets for embedding models."""

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample


class GliomaGloveDataset:
    def __init__(self, patient_gene_dict: dict) -> None:
        self._patient_gene_dict = patient_gene_dict
        self._tokens = []
        # get full list of tokens in the dataset
        for patient, genes in self._patient_gene_dict.items():
            for gene in genes:
                self._tokens.append(gene)

        # gene frequency counter
        word_counter = Counter()
        word_counter.update(self._tokens)
        # word to id and vice versa
        self._word2id = {
            w: i
            for i, (w, _) in enumerate(word_counter.most_common())
        }
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        # all tokens
        self._id_tokens = [self._word2id[w] for w in self._tokens]

        # dictionary with tokens
        self._patient_geneID_dict = {}
        for patient, genes in self._patient_gene_dict.items():
            self._patient_geneID_dict[patient] = [
                self._word2id[w] for w in genes
            ]

        self._create_coocurrence_matrix()

        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))
        print("Number of patients: {}".format(len(self._patient_gene_dict)))

    def _create_coocurrence_matrix(self):
        # will count the co-occurences of genes for the same patient only
        # NOTE this should be made clear in the paper that the patient
        # acts as the 'window' for sampling
        cooc_mat = defaultdict(Counter)
        for patient, genes in self._patient_geneID_dict.items():
            for w in genes:
                for c in genes:
                    if c != w:
                        cooc_mat[w][c] += 1

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self.cooc_mat = cooc_mat
        self._i_idx = torch.LongTensor(self._i_idx)
        self._j_idx = torch.LongTensor(self._j_idx)
        self._xij = torch.FloatTensor(self._xij)

    def get_batches(self, batch_size: int) -> torch.Tensor:
        #Generate random idx
        rand_ids = torch.LongTensor(
            np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p + batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[
                batch_ids]


class GliomaGene2VectDataset:
    def __init__(self, patient_gene_dict: dict) -> None:
        self._patient_gene_dict = patient_gene_dict
        self._tokens = []
        for patient, genes in self._patient_gene_dict.items():
            for gene in genes:
                self._tokens.append(gene)
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {
            w: i
            for i, (w, _) in enumerate(word_counter.most_common())
        }
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._id_tokens = [self._word2id[w] for w in self._tokens]

        # dictionary with tokens
        self._patient_geneID_dict = {}
        for patient, genes in self._patient_gene_dict.items():
            self._patient_geneID_dict[patient] = [
                self._word2id[w] for w in genes
            ]

        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))
        print("Number of patients: {}".format(len(self._patient_gene_dict)))

    def get_batches(self, batch_size: int) -> torch.LongTensor:

        target = []
        context = []
        batch = sample(self._patient_geneID_dict.items(), batch_size)
        for patient_genes in batch:
            try:
                samples = sample(patient_genes[1], k=2)
                target.append(samples[0])
                context.append(samples[1])
            except:
                continue

        yield torch.LongTensor(target), torch.LongTensor(context)
