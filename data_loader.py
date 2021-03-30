"""
This module contains all the necessary functions to create data loaders.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_mfccs(sample_mfcc, mfcc_mean, mfcc_std):
    """
    Function to normalize MFCCs data, according to mean and variance in train set.
    :param sample_mfcc: MFCC data to be normalized
    :param mfcc_mean: train set MFCC mean
    :param mfcc_std: train set MFCC stadard deviation
    :return: normalized MFCC data
    """
    return (sample_mfcc - mfcc_mean) / mfcc_std


def load_annotations(data_dir, mode):
    """
    Function to load the annotations.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :return: annotations as tensors
    """
    annotations = np.load(os.path.join(data_dir, '{:s}_annotations.npy'.format(mode)))
    # Convert numpy arrays to torch tensors
    annotations = torch.tensor(annotations.astype(np.float32))

    return annotations

def load_audio_data(data_dir, mode):
    """
    Function to load the audio features.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :return: audio features as tensors
    """
    data = np.load(os.path.join(data_dir, '{:s}_mfccs.npy'.format(mode)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_text_data(data_dir, mode, modality):
    """
    Function to load the lyrics/comments tokens, according to the `modality`.
    :param data_dir: path to data directory
    :param mode: type of annotations set to load - train | validation | test
    :param modality: select wether to load the lyrics or comments tokens
    :return: lyrics/comments tokens as tensors
    """
    data = np.load(os.path.join(data_dir, '{:s}_tokens_{:s}.npy'.format(mode, modality)))
    data = torch.tensor(data.astype(np.float32))

    return data

def load_embedding_matrix(data_dir, text_modality):
    """
    Function to load the embedding matrix with representations of lyrics/comments, according to `text_modality`.
    :param data_dir: path to data directory
    :param text_modality: select wether to load the lyrics or comments representations
    :return: embedding matrix as tensor
    """
    embedding_matrix = np.load(os.path.join(data_dir, 'embedding_matrix_{:s}.npy'.format(text_modality)))
    embedding_matrix = torch.tensor(embedding_matrix)

    return embedding_matrix


def make_audio_loader(data_dir, mode, mfcc_mean, mfcc_std, batch_size=64):
    """
    Function to create audio data loaders for training, validation or testing in batches with specified size.
    :param data_dir: path to data directory
    :param mode: type of data set to load - train | validation | test
    :param mfcc_mean: train set MFCC mean
    :param mfcc_std: train set MFCC stadard deviation
    :param batch_size: number of samples per training/testing batch
    :return: train/validation/test audio data loader
    """

    audio_data = load_audio_data(data_dir, mode)
    annotations = load_annotations(data_dir, mode)

    dataset_ = AudioDataset(audio_data, annotations, mfcc_mean, mfcc_std)
    dataloader_ = DataLoader(dataset_, batch_size=batch_size)

    return dataloader_


def make_text_loader(data_dir, mode, text_modality, batch_size=64):
    """
    Function to create audio data loaders for training, validation or testing in batches with specified size.
    :param data_dir: path to data directory
    :param mode: type of data set to load - train | validation | test
    :param text_modality: select wether to load the lyrics or comments data
    :param batch_size: number of samples per training/testing batch
    :return: train/validation/test audio data loader
    """
    text_data = load_text_data(data_dir, mode, text_modality)
    annotations = load_annotations(data_dir, mode)

    dataset_ = TextDataset(text_data, annotations)
    dataloader_ = DataLoader(dataset_, batch_size=batch_size)

    return dataloader_


class AudioDataset(Dataset):
    """
    This class contains custom definition for creating audio datasets.
    """
    def __init__(self, audio_data, annotations, mfcc_mean, mfcc_std):
        """
        :param audio_data: MFCCs data
        :param annotations: valence-arousal annotations
        :param mfcc_mean: train set MFCC mean
        :param mfcc_std: train set MFCC stadard deviation
        """
        self._audio_data = audio_data
        self._annotations = annotations

        self._mfcc_mean = mfcc_mean
        self._mfcc_std = mfcc_std

    def __len__(self):

        return len(self._annotations)

    def __getitem__(self, idx):

        # Get and normalize the MFCCs data at idx
        sample_mfcc = self._audio_data[idx]
        sample_mfcc = normalize_mfccs(sample_mfcc, self._mfcc_mean, self._mfcc_std)
        # Get the annotations for sample data at idx
        sample_annotations = self._annotations[idx]

        return sample_mfcc, sample_annotations


class TextDataset(Dataset):
    """
    This class contains custom definition for creating lyrics or comments datasets.
    """
    def __init__(self, text_data, annotations):
        """
        :param text_data: lyrics/comments tokens
        :param annotations: valence-arousal annotations
        """
        self._text_data = text_data
        self._annotations = annotations

    def __len__(self):

        return len(self._annotations)

    def __getitem__(self, idx):

        # Get tokens and annotations for sample data at idx
        sample_text = self._text_data[idx]
        sample_annotations = self._annotations[idx]

        return sample_text, sample_annotations
