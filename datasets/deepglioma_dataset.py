import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from datasets.data_utils import get_unk_mask_indices, image_loader


class DeepGlioma_Dataset(Dataset):
    def __init__(self,
                 num_labels: int,
                 data: Dict,
                 img_root: str,
                 transform: None,
                 known_labels: int = 0,
                 missing_value_marker: int = 2,
                 testing: bool = False) -> Dict:

        self.num_labels = num_labels
        self.data = data
        self.img_root = img_root
        self.transform = transform
        self.known_labels = known_labels
        self.missing_value_marker = missing_value_marker
        self.testing = testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image id and load image
        image_ID = self.data[idx]['file_name']
        img_name = os.path.join(self.img_root, image_ID)
        image = image_loader(img_name, self.transform)

        # get labels
        labels = self.data[idx]['objects']
        # convert UNK to arbitrary float value
        labels = list(
            map(lambda x: self.missing_value_marker
                if x == 'UNK' else x, labels))
        labels = torch.Tensor(labels)

        # get masks if training transformer model
        unk_mask_indices = get_unk_mask_indices(image, self.testing,
                                                self.num_labels,
                                                self.known_labels)
        mask = labels.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        # populate sample dictionary
        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample


class DeepGlioma_SSL(Dataset):
    def __init__(self, data: Dict, img_root: str, transform: None) -> Dict:

        self.data = data
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image id and load image
        image_ID = self.data[idx]['file_name']
        img_name = os.path.join(self.img_root, image_ID)
        image = image_loader(img_name, self.transform)

        # populate sample dictionary
        sample = {}
        sample['imageIDs'] = image_ID
        sample['image'] = image
        return sample
