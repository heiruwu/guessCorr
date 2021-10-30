from dataset.correlation import CorrelationDataset
from torchvision import transforms
from base import BaseDataLoader

from dataset.transforms.crop import CropAnd2GRAY
from dataset.transforms.to_tensor import ToTensor

class CorrelationDataLoader(BaseDataLoader):
    """
    Correlation data loader implementing BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True):
        trsfm = transforms.Compose([
            CropAnd2GRAY(),
            ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = CorrelationDataset(self.data_dir, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
