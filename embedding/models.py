"""Models for learning gene embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# GloVe Embedding Model
class GliomaGloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=2048):
        super(GliomaGloveModel, self).__init__()
        # Embedding layers for the context and target words
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

        # Initialize with uniform distribution
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x


# Gene2Vec Embedding Model
class GliomaGene2VecModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=2048,
                 device='cpu',
                 noise_dist=None,
                 negative_samples=5):
        super(GliomaGene2VecModel, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device
        self.noise_dist = noise_dist

        # Initialize with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1, 1)
        self.embeddings_context.weight.data.uniform_(-1, 1)

    def forward(self, input_word, context_word):

        emb_input = self.embeddings_input(input_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_input, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out_loss = F.logsigmoid(emb_product)

        if self.negative_samples > 0:
            # computing negative loss
            if self.noise_dist is None:
                noise_dist = torch.ones(self.vocab_size)
            else:
                noise_dist = self.noise_dist

            num_neg_samples_for_this_batch = context_word.shape[
                0] * self.negative_samples
            negative_example = torch.multinomial(
                noise_dist, num_neg_samples_for_this_batch, replacement=True)
            negative_example = negative_example.view(
                context_word.shape[0], self.negative_samples).to(self.device)
            emb_negative = self.embeddings_context(negative_example)
            emb_negative = self.embeddings_input(negative_example)
            emb_product_neg_samples = torch.bmm(emb_negative.neg(),
                                                emb_input.unsqueeze(2))
            noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(
                1)
            total_loss = -(out_loss + noise_loss).mean()
            return total_loss
        else:
            return -(out_loss).mean()
