import os
import torch
import pandas as pd
import cv2

from torch.utils.data import Dataset

class CorrelationDataset(Dataset):
    """
    Correlation dataset
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_file = os.path.join(data_dir, 'responses.csv')
        self.img_path = os.path.join(data_dir, 'images')
        self.correlations_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.correlations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_path,
                                '{}.png'.format(self.correlations_frame.iloc[idx, 0]))
        image = cv2.imread(img_name)

        correlations = self.correlations_frame.iloc[idx, 1]

        sample = {'image': image, 'correlations': correlations}

        if self.transform:
            sample = self.transform(sample)

        return sample
