import os
import numpy as np
import torch

from models import AudioNet, TextNet
from data_loader import make_audio_loader, make_text_loader, load_embedding_matrix
from utility_functions import *


class Tester:
    def __init__(self, args):

        self._data_dir = args.data_dir
        self._models_dir = args.models_dir
        self._plots_dir = args.plots_dir

        self._device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self._modality = args.modality

        if self._modality == 'audio':
            self.test_loader = make_audio_loader(self._data_dir, 'test', args.mfcc_mean, args.mfcc_std)
            self.model = AudioNet().to(self._device)
        else:
            self.test_loader = make_text_loader(self._data_dir, 'test', self._modality)
            self._embedding_matrix = load_embedding_matrix(self._data_dir, self._modality)
            self.model = TextNet(self._embedding_matrix).to(self._device)

    def load_model(self):

        model_path = os.path.join(self._models_dir, '{:s}_model.pt'.format(self._modality))
        self.model.load_state_dict(torch.load(model_path))

    def test(self):

        true_annotations = []
        pred_annotations = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, annotations) in enumerate(self.test_loader):

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
        valence_mae = np.mean(np.abs(true_valence - pred_valence))
        valence_mse = np.mean((true_valence - pred_valence) ** 2)

        valence_dict = {'true_annotations': true_valence,
                        'pred_annotations': pred_valence,
                        'mae': valence_mae,
                        'mse': valence_mse}

        true_arousal = np.array([annot[1] for annot in true_annotations])
        pred_arousal = np.array([annot[1] for annot in pred_annotations])
        arousal_mae = np.mean(np.abs(true_arousal - pred_arousal))
        arousal_mse = np.mean((true_arousal - pred_arousal) ** 2)

        arousal_dict = {'true_annotations': true_arousal,
                        'pred_annotations': pred_arousal,
                        'mae': arousal_mae,
                        'mse': arousal_mse}

        true_quadrant = np.array([get_quadrant(measurement) for measurement in true_annotations])
        pred_quadrant = np.array([get_quadrant(measurement) for measurement in pred_annotations])

        quadrants_dict = {'true_annotations': true_quadrant,
                          'pred_annotations': pred_quadrant}

        quadrants_names = [1, 2, 3, 4]
        for quadrant in quadrants_names:

            q_pred = true_quadrant[np.where(pred_quadrant == quadrant)]
            q_true = true_quadrant[np.where(true_quadrant == quadrant)]
            q_perc = np.sum(q_pred == quadrant) / len(q_true) * 100
            quadrants_dict[quadrant] = q_perc

        return valence_dict, arousal_dict, quadrants_dict
