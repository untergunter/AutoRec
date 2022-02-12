import torch
import torch.nn as nn
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
        super(VarAutoRec, self).__init__(number_of_items=number_of_items,
                                         hidden_size=hidden_size,
                                         activation_function_1=activation_function_1,
                                         activation_function_2=activation_function_2,
                                         loss=loss,
                                         λ=λ,
                                         lt=lr)

        self.log_var_layer = nn.Linear(hidden_size, 2)
        self.mean_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.encoder(x)
        encoded = self.act_1(out)

        encoded_mean, encoded_var = self.mean_layer(encoded), self.log_var_layer(encoded)
        eps = torch.randn(encoded_mean.size(0), encoded_mean.size(1)).to(encoded_mean.get_device())
        out = encoded_mean + eps * torch.exp(encoded_var/2)

        out = self.decoder(out)
        out = self.act_2(out)
        return encoded, encoded_mean, encoded_var, out

    def training_step(self, train_batch, batch_idx):
        x, y, y_mask = train_batch
        encoded, encoded_mean, encoded_var, y_hat = self.forward(x)

        # set to 0 unseen by users
        y_hat *= y_mask
        y *= y_mask

        kl_div = -0.5*(torch.sum(1 + encoded_var - encoded_mean**2 - torch.exp(encoded_var), axis=1))

        loss = torch.sum(self.loss_func(y_hat, y)*y_mask)
        self.log('train_loss', loss)
        return loss