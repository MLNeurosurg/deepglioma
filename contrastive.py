"""Module for training SRH vision encoder using weakly-supervised, patch-based, 
mulit-label contrastive learning, or patchcon. 
"""

import datetime
from typing import TextIO
import json
import yaml
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import dataset
from datasets.deepglioma_dataset import DeepGlioma_Dataset
from datasets.data_utils import label_frequency, oversample_label
from datasets.data_utils import image_transforms, get_glioma_data, get_labels

# import models
from models.losses import SupConLoss
from models.supcon_trainer import PatchConTrainer
from models.utils import save_model

# training modules
from runners import run_epoch_patchcon

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


#### DATA PREP #################################################################
# load the training data
labels = config_dict['data']['labels']

gen_series = pd.DataFrame()
gen_labels = pd.DataFrame()
for center in config_dict['data']['train_centers']:
    gen_series = gen_series.append(
        pd.read_excel(config_dict['data']['data_spreadsheet'],
                      sheet_name=f'{center}_series'),
        ignore_index=True)
    gen_labels = gen_labels.append(pd.read_excel(
        config_dict['data']['data_spreadsheet'], sheet_name=f'{center}_data'),
                                   ignore_index=True)
gen_labels = get_labels(gen_labels, labels)

# generate training dataset
train_data = get_glioma_data(study_df=gen_series,
                       labels_df=gen_labels,
                       data_root_path=config_dict['data']['data_root'])

# balance the labels to improve training
if len(labels) == 1:
    train_data = oversample_label(train_data,
                                  label_index=0,
                                  perc_majority_label=1)
if len(labels) >= 3:
    train_data = oversample_label(train_data,
                                  label_index=2,
                                  perc_majority_label=1)
    train_data = oversample_label(train_data,
                                  label_index=1,
                                  perc_majority_label=0.5)
train_freq, label_array = label_frequency(train_data)
print(f'Training label frequency: {train_freq}')

# define the label weight vector
label_weights = config_dict['training']['label_weights']
weights = torch.zeros(size=(1, len(label_weights)))
for i, (label, weight) in enumerate(label_weights.items()):
    weights[:, i] = weight

#### GET DATALOADERS ###########################################################
# train dataloader
train_dataset = DeepGlioma_Dataset(num_labels=len(labels),
                                   data=train_data,
                                   img_root=config_dict['data']['data_root'],
                                   transform=image_transforms(
                                       image_size=config_dict['data']['image_size'],
                                       strength=config_dict['data']['transform_strength']),
                                   known_labels=0,
                                   testing=False)
train_loader = DataLoader(train_dataset, 
                        batch_size=config_dict['training']['batch_size'], 
                        shuffle=True)

#### LOAD MODELS ###############################################################
# Instantiate the supervised contrastive trainer wrapper
model = PatchConTrainer(num_labels=len(labels),
                      backbone=config_dict['model']['vision_backbone'],
                      input_size=config_dict['model']['input_size'],
                      projector_dim=config_dict['model']['projector_dim'],
                      pretrained=True,
                      pretrained_vision_encoder=config_dict['model']
                      ['pretrained_vision_model'])
# distribute model on GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model = model.cuda()
print(model)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=config_dict['training']['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                    T_max=config_dict['training']['n_epochs'])

# instantiate loss function
supcon_loss = SupConLoss(temperature=config_dict['training']['temp'])

#### TRAIN MODELS ##############################################################
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'

# initialize dictionaries to store results
train_losses = {}
for epoch in range(0, config_dict['training']['n_epochs'] + 1):
    print(f'======================== {epoch} ========================')
    ################### Train ##################################################
    loss_total, epoch_losses, all_image_ids = run_epoch_patchcon(
        model,
        train_loader,
        supcon_loss,
        optimizer,
        scheduler,
        n_labels=len(labels),
        label_weights=weights,
        iterations=config_dict['training']['iters'],
        train=True)

    train_losses[epoch] = epoch_losses
    if epoch % 5 == 0:
        print(f'Saving model for epoch {epoch}')
        # save the Vision_Encoder model only
        save_model(
            model.module.vision_encoder,
            f'{config_dict["model"]["vision_backbone"]}_{epoch}_{np.round(loss_total, decimals=2)}_{now}_con_model'
        )
