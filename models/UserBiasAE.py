import torch
from models.AutoRecBase import AutoRecBase


class UserBiasAE(AutoRecBase):

    def forward(self, x):
        user_bias = x.nanmean() if len(x.shape) == 1 else x.nanmean(dim=1)
        x = torch.nan_to_num(x, 0)
        out = super().forward(x)
        out = (user_bias + out.T).T
        return out
