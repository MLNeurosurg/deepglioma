"""Main script for pretraining the genetic embedding layer."""

import pandas as pd
import numpy as np
from typing import TextIO
import json
import yaml
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

# import modules
from models import GliomaGene2VecModel, GliomaGloveModel
from datasets import GliomaGloveDataset, GliomaGene2VectDataset
from utils import binary_to_string, convert_df_to_dict, weight_func

def wmse_loss(weights: torch.Tensor, inputs: torch.Tensor,
              targets: torch.Tensor) -> torch.Tensor:
    """Weighted mean sqaured error loss for GloVe model."""
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)

# get config file
def parse_args() -> TextIO:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config
cf_fd = parse_args()

def main():
    cmd_input: TextIO = parse_args()
    if cmd_input.name.endswith(".json"):
        return json.load(cf_fd)
    elif cmd_input.name.endswith(".yaml") or cf_fd.name.endswith(".yml"):
        return yaml.load(cf_fd, Loader=yaml.FullLoader)
config_dict = main()

# specify model type: glove or gene2vec
model_type = config_dict['model']['model_type']

# import dataframe with genetic data
embedding_df = pd.read_excel(config_dict['data']['data_spreadsheet'])
embedding_df.drop(['Tumor Type'], axis=1, inplace=True)
embedding_df = embedding_df[['Sample ID'] + config_dict['data']['mutations']]
embedding_df['idh'] = embedding_df['idh'].astype("float")
embedding_df.dropna(how='all', inplace=True)
print(embedding_df.head(10))

# formatting and conversion to dictionary
embedding_df = binary_to_string(embedding_df)
embedding_dict = convert_df_to_dict(embedding_df)

# import model and dataset
if model_type == 'glove':
    dataset = GliomaGloveDataset(embedding_dict)
    model = GliomaGloveModel(vocab_size=dataset._vocab_len,
                                embedding_dim=config_dict['model']['embedding_dim'])
else:
    dataset = GliomaGene2VectDataset(embedding_dict)
    model = GliomaGene2VecModel(vocab_size=dataset._vocab_len,
                                 embedding_dim=config_dict['model']['embedding_dim'])

# run training
optimizer = optim.Adam(model.parameters(), lr=config_dict['training']['lr'])
loss_values = list()
for e in range(1, config_dict['training']['n_epochs'] + 1):
    # forward pass
    if model_type == 'glove':
        for x_ij, i_idx, j_idx in dataset.get_batches(config_dict['training']['batch_size']):

            # forward pass
            optimizer.zero_grad()
            outputs = model(i_idx, j_idx)
            weights_x = weight_func(x_ij, 
                                    config_dict['training']['x_max'], 
                                    config_dict['training']['alpha'])
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
            
            # backward pass
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

    else:
        for context, target in dataset.get_batches(config_dict['training']['batch_size']):
            
            # forward pass
            optimizer.zero_grad()
            loss = model(context, target)
            
            # backward pass
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

    print(f"Epoch {e} >>> Loss: {np.mean(loss_values[-20:])}")
print("Saving model...")
torch.save(model.state_dict(),
            f"glioma_{model_type}_embedding_model.pt")
