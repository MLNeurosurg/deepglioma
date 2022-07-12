"""Module with Classifier models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from .utils import freeze_weights, load_model
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT

from .encoder import Vision_Encoder
from .transformer_layers import SelfAttnLayer
from .utils import custom_replace, weights_init, freeze_weights
from .position_enc import positionalencoding2d


class Linear_Classifier(nn.Module):
    """Standard multilabel binary relevance model.
	backbone can be 'resnet50', 'inception', 'resnext', 'vit_s', 'vit_b', 'vit_l'."""
    def __init__(self,
                 num_labels: int = 3,
                 backbone: str = "resnet",
                 pretrained_vision_encoder: str = '',
                 freeze_backbone: bool = False,
                 image_size: int = 300):
        super(Linear_Classifier, self).__init__()

        self.num_labels = num_labels

        self.vision_encoder = Vision_Encoder(backbone=backbone,
                                             pretrained=True,
                                             image_size=image_size)

        # load pretrained model if needed
        if len(pretrained_vision_encoder) > 0:
            print('You are loading a pretrained model.')
            self.vision_encoder = load_model(self.vision_encoder,
                                             pretrained_vision_encoder)

        # freeze CNN feature extractor weights
        if freeze_backbone:
            self.vision_encoder = freeze_weights(self.vision_encoder)

        if "vit" in backbone:
            self.classifier = torch.nn.Linear(2048, self.num_labels)
        else:
            # self.classifier = nn.Linear(self.vision_encoder.fc.in_features, self.num_labels)
            self.classifier = nn.Linear(2048, self.num_labels)

    def forward(self, images, masks=None):
        features = self.vision_encoder(images)
        logits = self.classifier(features)
        return logits, features


class Tran_Classifier(nn.Module):
    def __init__(self,
                 num_labels: int,
                 image_size: int = 300,
                 backbone: str = 'resnet',
                 embedding_dim: int = 2048,
                 pretrained_label_embedding: torch.Tensor = None,
                 freeze_embedding: bool = True,
                 pretrained_vision_encoder: str = '',
                 freeze_backbone: bool = False,
                 use_lmt: bool = True,
                 pos_emb: bool = False,
                 layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1):

        super(Tran_Classifier, self).__init__()
        self.use_lmt = use_lmt

        # CNN backbone
        self.vision_encoder = Vision_Encoder(backbone=backbone,
                                             pretrained=True,
                                             image_size=image_size)

        # load pretrained model if needed
        if len(pretrained_vision_encoder) > 0:
            print('You are loading a pretrained model.')
            self.vision_encoder = load_model(self.vision_encoder,
                                             pretrained_vision_encoder)

        # freeze CNN feature extractor weights
        if freeze_backbone:
            self.vision_encoder = freeze_weights(self.vision_encoder)

        # Label embedding
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,
                                                                    -1).long()
        self.label_lt = torch.nn.Embedding(num_labels,
                                           embedding_dim,
                                           padding_idx=None)
        if pretrained_label_embedding is not None:
            assert num_labels == pretrained_label_embedding.shape[0]
            assert embedding_dim == pretrained_label_embedding.shape[1]
            self.label_lt = self.label_lt.from_pretrained(
                pretrained_label_embedding, freeze=freeze_embedding)

        if freeze_embedding:
            self.label_lt = freeze_weights(self.label_lt)

        # state embeddings
        self.known_label_lt = torch.nn.Embedding(3,
                                                 embedding_dim,
                                                 padding_idx=0)

        # Positional encoding
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            self.position_encoding = positionalencoding2d(
                embedding_dim, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([
            SelfAttnLayer(embedding_dim, heads, dropout) for _ in range(layers)
        ])
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.output_linear = torch.nn.Linear(embedding_dim, num_labels)
        self.projector = torch.nn.Linear(embedding_dim, embedding_dim)

        # initialize weights
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.layernorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, mask=None):

        # initialize a label vector
        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        # feedfoward CNN pass and normalize features
        features = self.vision_encoder(images)
        features = F.normalize(features, p=2, dim=1)  # ADDED Line

        if self.use_pos_enc:
            pos_encoding = self.position_encoding(
                features,
                torch.zeros(features.size(0), 18, 18, dtype=torch.bool).cuda())
            features = features + pos_encoding
        features = features.view(features.size(0), features.size(1),
                                 -1).permute(0, 2, 1)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()
            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)
            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        # concatenate image features and label embeddings
        embeddings = torch.cat((features, init_label_embeddings), 1)
        embeddings = self.layernorm(embeddings)

        # get attention values
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # get label embeddings/exclude visual embedding
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]

        # get label embedding weights and batch matrix multiplication
        label_embedding_weights = self.label_lt.weight.t().unsqueeze(0).repeat(
            label_embeddings.size(0), 1, 1).detach()
        mat_logits = torch.bmm(label_embeddings, label_embedding_weights)

        # select the main diagonal to get the correct logits
        logits = torch.diagonal(mat_logits, dim1=-2, dim2=-1)

        return logits, features
