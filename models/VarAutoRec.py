import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.AutoRecBase import AutoRecBase


class VarAutoRec(AutoRecBase):
    def forward(self, x):
        x = super().forward(x)

