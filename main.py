"""DeepGlioma for rapid molecular classification of fresh brain tumor specimens
imaged using stimulated Raman histology. This is the main script for training and
validating models.
"""

import datetime
from typing import TextIO
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# import dataset
from datasets.deepglioma_dataset import DeepGlioma_Dataset
from datasets.data_utils import train_validation_split, label_frequency, oversample_label
from datasets.data_utils import image_transforms, get_glioma_data, get_labels

# import models
from models.classifiers import Linear_Classifier, Tran_Classifier
from models.utils import get_embedding_weights, save_model

# training modules
from runners import run_epoch
from utils.evaluate import compute_metrics, print_results
from utils.metrics import aggregate_predictions


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
print(gen_labels)

# load validation data
if config_dict['validation']['val_type'] == 'easy':
    validation_cases = [
        "NIO_UM_072", "NIO_UM_158", "NIO_UM_298", "NIO_UM_318", "NIO_UM_492",
        "NIO_UM_503", "NIO_UM_575", "NIO_UM_655"
    ]
if config_dict['validation']['val_type'] == 'hard':
    validation_cases = [
        'NIO_UM_513', 'NIO_UM_644', 'NIO_UM_744', 'NIO_UM_747', 'NIO_UM_824',
        'NIO_UM_864', 'NIO_UM_883', 'NIO_UM_929'
    ]
if config_dict['validation']['val_type'] == 'random':
    validation_cases = gen_labels.groupby("idh").sample(
        n=10, random_state=config_dict['validation']
        ['random_seed'])['dbcase'].tolist()
if config_dict['validation']['val_type'] == None:
    validation_cases = []
print(f'Validation cases: {sorted(validation_cases)}')

# generate dataset
data = get_glioma_data(study_df=gen_series,
                       labels_df=gen_labels,
                       data_root_path=config_dict['data']['data_root'])
train_data, val_data = train_validation_split(data, validation_cases)

# balance the labels to improve training
if len(labels) is 1:
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

if config_dict['validation']['val_type'] is not None:
    val_freq, label_array = label_frequency(val_data)
    print(f'Validation label frequency: {val_freq}')

# compute label weights matrix
if config_dict['training']['label_weights']:
    weights = np.zeros(shape=(2, len(config_dict['data']['labels'])))
    for l in range(len(labels)):
        weights[:, l] = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(label_array[:,
                                                                           l]),
                                             y=label_array[:, l])
else:
    weights = None

#### GET DATALOADERS ###########################################################
# train dataloader
train_dataset = DeepGlioma_Dataset(
    num_labels=len(labels),
    data=train_data,
    img_root=config_dict['data']['data_root'],
    transform=image_transforms(
        image_size=config_dict['data']['image_size'],
        strength=config_dict['data']['transform_strength']),
    known_labels=config_dict['training']['known_labels'],
    testing=False)
train_loader = DataLoader(train_dataset,
                          batch_size=config_dict['training']['batch_size'],
                          shuffle=True)

# validation dataloader
val_transforms = transforms.Compose([transforms.ToTensor()])
val_dataset = DeepGlioma_Dataset(num_labels=len(labels),
                                 data=val_data,
                                 img_root=config_dict['data']['data_root'],
                                 transform=val_transforms,
                                 known_labels=0,
                                 testing=False)
valid_loader = DataLoader(val_dataset,
                          batch_size=config_dict['training']['batch_size'],
                          shuffle=False)

#### LOAD MODELS ###############################################################
# All models currently using ResNet50
classifier = config_dict['model']['classifier']
if classifier == 'linear':
    model = Linear_Classifier(
        num_labels=len(labels),
        backbone=config_dict['model']['vision_backbone'],
        pretrained_vision_encoder=config_dict['model']
        ['pretrained_vision_model'],
        freeze_backbone=config_dict['model']['freeze_feature_extractor'])

if classifier == 'tran':
    if config_dict['model']['pretrain_embed']:
        embed_weights = torch.load(config_dict['model']['embed_weights'])
        gene_to_idx = torch.load(config_dict['model']['gene_to_idx'])
        emb_weights = get_embedding_weights(embed_weights,
                                            config_dict['data']['labels'],
                                            gene_to_idx)

    else:
        emb_weights = None
    model = Tran_Classifier(
        num_labels=len(labels),
        backbone=config_dict['model']['vision_backbone'],
        embedding_dim=2048,
        pretrained_label_embedding=emb_weights,
        freeze_embedding=config_dict['model']['freeze_embedding'],
        pretrained_vision_encoder=config_dict['model']
        ['pretrained_vision_model'],
        freeze_backbone=config_dict['model']['freeze_feature_extractor'],
        use_lmt=True,
        pos_emb=False,
        layers=3,
        heads=8,
        dropout=config_dict['training']['dropout'])

# distribute model on GPUs
model_parallel = False
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
    model_parallel = True
model = model.cuda()
print(model)

# OPTIMIZER
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config_dict['training']['lr'],
    weight_decay=config_dict['training']['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config_dict['training']['n_epochs'])

#### TRAIN MODELS ##############################################################
# mark the time for saving results and models
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'

# initialize dictionaries to store results
train_metrics = {}
val_predictions = {}
val_metrics = {}
total_correct_tracker = {}
total_correct_tracker[0] = 0
for epoch in range(1, config_dict['training']['n_epochs'] + 1):
    print(f'======================== {epoch} ========================')
    ################### Train ##################################################
    all_preds, all_targs, all_masks, all_missing_masks, all_ids, train_loss, train_loss_unk = run_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        n_labels=len(labels),
        label_weights=weights,
        iterations=config_dict['training']['iters'],
        train=True,
        model_type=classifier)
    train_metrics = compute_metrics(
        all_preds,
        all_targs,
        all_masks,
        all_missing_masks,
        train_loss,
        train_loss_unk,
        known_labels=config_dict['training']['known_labels'],
        missing_values=config_dict['training']['missing_values'])
    train_metrics[epoch] = train_metrics
    print(train_metrics)
    if config_dict['validation']['val_type'] == None:
        continue
    ################### Validate ###############################################
    all_preds, all_targs, all_masks, all_missing_masks, all_ids, valid_loss, valid_loss_unk = run_epoch(
        model,
        valid_loader,
        optimizer=None,
        scheduler=None,
        n_labels=len(labels),
        train=False,
        model_type=classifier)
    valid_epoch_metrics = compute_metrics(
        all_preds,
        all_targs,
        all_masks,
        all_missing_masks,
        valid_loss,
        valid_loss_unk,
        known_labels=0,
        missing_values=config_dict['training']['missing_values'])
    aggregation_dict, correct_dict, binary_pred_dict, label_dict = aggregate_predictions(
        all_ids,
        all_preds,
        all_targs,
        all_masks,
        all_missing_masks,
        prediction_level='patient')

    # print and log aggregation statistics
    total_correct = print_results(labels, correct_dict, binary_pred_dict,
                                  label_dict)

    # populate epoch dictionary
    valid_epoch_predictions = {}
    val_preds = (all_preds, all_targs, all_masks, all_missing_masks, all_ids,
                 valid_loss, valid_loss_unk)
    names = [
        'all_preds', 'all_targs', 'all_masks', 'all_missing_masks', 'all_ids',
        'valid_loss', 'valid_loss_unk'
    ]
    for name, val in zip(names, val_preds):
        valid_epoch_predictions[name] = val

    # store results
    val_predictions[epoch] = valid_epoch_predictions
    val_metrics[epoch] = valid_epoch_metrics

    # save model and checkpointing
    print(f'Saving model for epoch {epoch}')
    save_model(
        model,
        f'{config_dict["model"]["vision_backbone"]}_{config_dict["model"]["classifier"]}_{config_dict["validation"]["val_type"]}_{epoch}_{np.round(total_correct, decimals=3)}_{now}_model',
        data_parallel=model_parallel)
    print(valid_epoch_metrics)

# save validation data
torch.save(
    val_predictions,
    f'{config_dict["model"]["vision_backbone"]}_{config_dict["model"]["classifier"]}_{config_dict["validation"]["val_type"]}_{now}_val_predictions.pt'
)
torch.save(
    val_metrics,
    f'{config_dict["model"]["vision_backbone"]}_{config_dict["model"]["classifier"]}_{config_dict["validation"]["val_type"]}_{now}_metrics.pt'
)
