import torch
import torch.nn as nn

from base import BaseModel


class CorrelationModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2), # (3*3*1 +1) * 16 = 160
            nn.BatchNorm2d(16), # 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=2), # 3*3*16*32 + 32 = 4640
            nn.BatchNorm2d(32), # 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 3*3*32*64 + 64 = 18496
            nn.BatchNorm2d(64), # 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # 3*3*64*32 + 32 = 18464
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 3*3*32*16 + 16 = 4624
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 3*3*16*8 + 8 = 4624
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1, padding=0), # 1*1*16+1 = 17
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.linear(x)
        x = torch.flatten(x, 1)
        return x
