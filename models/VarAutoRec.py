import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.AutoRecBase import AutoRecBase


class VarAutoRec(AutoRecBase):

    def __init__(self,
                 number_of_items: int,
                 hidden_size: int,
                 activation_function_1,
                 activation_function_2,
                 loss,
                 λ=0.01,
                 lr=0.001):
        super(VarAutoRec, self).__init__()

        self.encoder = nn.Linear(number_of_items, hidden_size)
        self.act_1 = activation_function_1()
        self.log_var_layer = nn.Linear(hidden_size, 2)

        self.mean_layer = nn.Linear(hidden_size, 2)
        self.decoder = nn.Linear(hidden_size, number_of_items)

        self.act_2 = activation_function_2()
        self.loss_func = loss()
        self.λ = λ,
        self.lr = lr

    def forward(self, x):
        out = self.encoder(x)
        out = self.act_1(out)

        out_mean, out_var = self.mean_layer(out), self.log_var_layer(out)
        eps = torch.randn(out_mean.size(0), out_mean.size(1)).to(out_mean.get_device())
        out = out_mean + eps * torch.exp(out_var/2)

        out = self.decoder(out)
        out = self.act_2(out)
        return out

