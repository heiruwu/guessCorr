import os
import torch
import pandas as pd
import cv2
import numpy as np

from torch.utils.data import Dataset

class CorrelationDataset(Dataset):
    """
    Correlation dataset
    """

    def __init__(self, data_dir, transform=None, test_split=0.2, training=True):
        """
        Args:
            data_dir (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            training (boolean, optional): Training or not, default to True
        """
        csv_file = os.path.join(data_dir, 'responses.csv')
        self.img_path = os.path.join(data_dir, 'images')
        self.correlations_frame = pd.read_csv(csv_file)
        self.transform = transform

        self.dataset = self._split_dataset(test_split, training)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_path,
                                '{}.png'.format(self.dataset.iloc[idx, 0]))
        image = cv2.imread(img_name)

        correlations = self.dataset.iloc[idx, 1]

        sample = {'image': image, 'correlations': correlations}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _split_dataset(self, split, training=True):
        if split == 0.0:
            return None, None

        n_samples = len(self.correlations_frame)

        idx_full = np.arange(n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < n_samples, "testing set size is configured to be larger than entire dataset."
            len_test = split
        else:
            len_test = int(n_samples * split)

        test_idx = idx_full[0:len_test]
        train_idx = np.delete(idx_full, np.arange(0, len_test))

        if training:
            dataset = self.correlations_frame.ix[train_idx]
        else:
            dataset = self.correlations_frame.ix[test_idx]

        return dataset
