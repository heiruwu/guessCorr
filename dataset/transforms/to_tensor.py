import numpy as np
import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, correlations = sample['image'], sample['correlations']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        correlations = np.array([correlations])
        return {'image': torch.from_numpy(image).float(),
                'correlations': torch.from_numpy(correlations).float()}
