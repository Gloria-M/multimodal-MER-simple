import os
import numpy as np
import torch
import torch.nn as nn

from models import AudioNet, TextNet
from data_loader import make_audio_loader, make_text_loader, load_embedding_matrix
from utility_functions import *


class Trainer:
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir
        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        self._lr = args.lr_init
        self._lr_decay = args.lr_decay
        self._weight_decay = args.weight_decay

        self._modality = args.modality

        if self._modality == 'audio':
            self.train_loader = make_audio_loader(self._data_dir, 'train', args.mfcc_mean, args.mfcc_std)
            self.validation_loader = make_audio_loader(self._data_dir, 'validation', args.mfcc_mean, args.mfcc_std)
            self.model = AudioNet().to(self._device)
        else:
            self.train_loader = make_text_loader(self._data_dir, 'train', self._modality)
            self.validation_loader = make_text_loader(self._data_dir, 'validation', self._modality)
            self._embedding_matrix = load_embedding_matrix(self._data_dir, self._modality)
            self.model = TextNet(self._embedding_matrix).to(self._device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        self._criterion = nn.MSELoss()

        self.train_dict = {'valence_loss': [], 'arousal_loss': []}
        self.validation_dict = {'valence_loss': [], 'arousal_loss': []}

    def save_model(self):

        model_path = os.path.join(self._models_dir, '{:s}_model.pt'.format(self._modality))
        torch.save(self.model.state_dict(), model_path)

    def update_learning_rate(self):

        self._lr *= self._lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr

        success_message = 'Learning rate updated to {:.1e}'.format(self._lr)
        print(success_format(success_message))

    def train(self):

        true_annotations = []
        pred_annotations = []

        self.model.train()
        for batch_idx, (data, annotations) in enumerate(self.train_loader):

            if self._modality != 'audio':
                data = data.long()

            data = data.to(self._device)
            annotations = annotations.to(self._device)

            self.optimizer.zero_grad()
            output = self.model(data)

            true_annotations.extend(annotations.cpu().detach().numpy())
            pred_annotations.extend(output.cpu().detach().numpy())

            batch_loss = self._criterion(output, annotations)
            batch_loss.backward()

            self.optimizer.step()

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.train_dict['valence_loss'].append(valence_mse)
        self.train_dict['arousal_loss'].append(arousal_mse)

    def validate(self):

        true_annotations = []
        pred_annotations = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.validation_loader):

                if self._modality != 'audio':
                    data = data.long()

                data = data.to(self._device)
                annotations = annotations.to(self._device)

                output = self.model(data)

                true_annotations.extend(annotations.cpu().detach().numpy())
                pred_annotations.extend(output.cpu().detach().numpy())

        true_annotations = np.array(true_annotations)
        pred_annotations = np.array(pred_annotations)

        true_valence = np.array([annot[0] for annot in true_annotations])
        pred_valence = np.array([annot[0] for annot in pred_annotations])
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        self.validation_dict['valence_loss'].append(valence_mse)
        self.validation_dict['arousal_loss'].append(arousal_mse)
