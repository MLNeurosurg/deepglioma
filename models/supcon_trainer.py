"""Module with Multi-Label Supervised Contrastive Learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifiers import Vision_Encoder
from .utils import load_model


class PatchConTrainer(nn.Module):
    def __init__(self,
                 num_labels: int = 3,
                 backbone: str = "resnet",
                 input_size: int = 2048,
                 projector_dim: int = 128,
                 pretrained: bool = True,
                 pretrained_vision_encoder: str = ''):
        super(PatchConTrainer, self).__init__()

        self.num_labels = num_labels

        # CNN backbone
        self.vision_encoder = Vision_Encoder(backbone=backbone,
                                             pretrained=pretrained)

        # load pretrained model if needed
        if len(pretrained_vision_encoder) > 0:
            print('You are loading a pretrained model.')
            self.vision_encoder = load_model(self.vision_encoder,
                                             pretrained_vision_encoder)

        projectors = {}
        for i in range(self.num_labels):
            projectors[f'label_projector_{i}'] = nn.Linear(
                input_size, projector_dim)

        self.projectors = nn.ModuleDict(projectors)

    def forward(self, images):

        features = self.vision_encoder(images)

        # normalize the representations
        features = F.normalize(features, p=2, dim=1)

        # reshape the features
        features = features.view(features.size(0), features.size(1),
                                 -1).permute(0, 2, 1)

        outputs = []
        for i in range(self.num_labels):
            preds = self.projectors[f'label_projector_{i}'](features)
            # normalize the projections
            preds = F.normalize(preds, p=2, dim=2)
            outputs.append(preds)

        return outputs, features
