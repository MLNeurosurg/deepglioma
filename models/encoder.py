"""Module with Vision Encoder Model."""

import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT


class Vision_Encoder(nn.Module):
    """Vision Encoder module for general feature extraction.
	backbone can be 'resnet50', 'inception', 'resnext', 'vit_s', 'vit_b', 'vit_l'."""
    def __init__(self,
                 backbone: str = "resnet",
                 pretrained: bool = True,
                 image_size: int = 300):
        super(Vision_Encoder, self).__init__()

        # Instantiate model
        if backbone == "resnet":
            self.vision_encoder = models.resnet50(pretrained=pretrained)
        if backbone == "inception":
            self.vision_encoder = models.inception_v3(pretrained=pretrained)
        if backbone == "resnext":
            self.vision_encoder = models.resnext50_32x4d(pretrained=pretrained)
        if 'vit' in backbone:
            if backbone == 'vit_s':
                layers, heads = 6, 16
            if backbone == 'vit_b':
                layers, heads = 12, 16
            if backbone == 'vit_l':
                layers, heads = 24, 16
            self.vision_encoder = ViT(image_size=image_size,
                                      patch_size=30,
                                      num_classes=1,
                                      dim=2048,
                                      depth=layers,
                                      heads=heads,
                                      mlp_dim=1024,
                                      pool='cls',
                                      dropout=0.1)
            print('You are using a ViT model.')

        # set final layer to identity
        if "vit" in backbone:
            self.vision_encoder.mlp_head[1] = nn.Identity()
        else:
            self.vision_encoder.fc = nn.Identity()

    def forward(self, images):
        features = self.vision_encoder(images)
        return features
