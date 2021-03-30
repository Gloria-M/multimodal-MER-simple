"""
This module contains the definitions of Neural Network models used for training on audio or text features.
"""
import torch
import torch.nn as nn


class AudioNet(nn.Module):
    """
    This class contains the definition of the architecture and the forward-pass of the model to be trained
     on audio features.
    """
    def __init__(self):
        super().__init__()

        in_ch = 20
        num_filters1 = 16
        num_filters2 = 16
        num_hidden = 34
        out_size = 2

        self._conv1 = nn.Sequential(nn.Conv1d(in_ch, num_filters1, 10, 1),
                                    nn.BatchNorm1d(num_filters1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(4, 4))
        self._conv2 = nn.Sequential(nn.Conv1d(num_filters1, num_filters2, 10, 1),
                                    nn.BatchNorm1d(num_filters2),
                                    nn.ReLU(),
                                    nn.MaxPool1d(4, 4))
        self._pool = nn.AvgPool1d(4, 4)
        self._drop = nn.Dropout(0.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(22*16, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        x = self._drop(x)

        x = x.view(-1, 22*16)

        x = self._fc1(x)
        x = self._drop(x)
        x = self._act(x)
        x = self._fc2(x)

        return x


class TextNet(nn.Module):
    """
    This class contains the definition of the architecture and the forward-pass of the model to be trained
     on lyrics or comments features.
    """
    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, num_feats = embedding_matrix.shape
        in_ch = 1
        num_filters = 16
        num_hidden = 32
        out_size = 2

        # Load weights and freeze gradients for the embedding layer
        self._emb = nn.Embedding.from_pretrained(embedding_matrix)

        self._conv = nn.Sequential(nn.Conv2d(in_ch, num_filters, (10, num_feats), 1),
                                   nn.BatchNorm2d(num_filters),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2, 1), 2),)
        self._pool = nn.AvgPool1d(4, 4)
        self._drop = nn.Dropout(0.5)
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(16 * 61, num_hidden)
        self._fc2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        x = self._emb(x)
        x = torch.unsqueeze(x, 1)
        x = x.float()

        x = self._conv(x)
        x = torch.squeeze(x)
        x = self._pool(x)

        x = x.view(-1, 16 * 61)
        x = self._drop(x)
        x = self._fc1(x)
        x = self._drop(x)
        x = self._fc2(x)

        return x

    def freeze_embedding_layer(self):
        """
        Method to freeze the gradients for the embedding layer.
        """
        self._emb.weight.requires_grad = False
