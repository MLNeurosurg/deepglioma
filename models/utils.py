import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_embedding_weights(embedding_weights, mutations, gene_idx_dict):
    # find mutations indices
    gene_idx = [
        idx for gene, idx in gene_idx_dict.items()
        if gene.split('mut')[0] in mutations
    ]
    emb_i = embedding_weights['wi.weight'][gene_idx]
    emb_j = embedding_weights['wj.weight'][gene_idx]
    # average i and j embedding vectors from glove model
    emb = (emb_i + emb_j) / 2
    return emb


def freeze_weights(model):
    """Freeze model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    return model


def save_model(model, model_name, data_parallel=False):
    """Save trained models."""
    if data_parallel:
        torch.save(model.module.state_dict(), model_name + '.pt')
    else:
        torch.save(model.state_dict(), model_name + '.pt')


def load_model(model_class, model_path):
    """Load previously trained models."""
    model_class.load_state_dict(torch.load(model_path), strict=True)
    return model_class


def custom_replace(tensor, on_neg_1, on_zero, on_one=1):
    res = tensor.clone()
    res[tensor == -1] = on_neg_1
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res


def custom_mask_missing(tensor, missing_value=2):
    res = tensor.clone()
    res[tensor == missing_value] = 0
    res[tensor != missing_value] = 1
    return res


def init_weights(m):
    print('Xavier Init')
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


def weights_init(module):
    """Initialize CTran weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)