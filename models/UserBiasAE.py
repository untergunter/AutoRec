import torch
import torch.nn as nn
import pytorch_lightning as pl

class UserBiasAE(pl.LightningModule):
    def __init__(self,
                 number_of_items: int,
                 hidden_size: int,
                 activation_function_1,
                 activation_function_2,
                 loss):
        super(UserBiasAE, self).__init__()

        self.encoder = nn.Linear(number_of_items, hidden_size)
        self.act_1 = activation_function_1()
        self.decoder = nn.Linear(hidden_size, number_of_items)
        self.act_2 = activation_function_2()
        self.loss_func = loss()

    def forward(self, x):
        user_bias = x.nanmean() if len(x.shape)==1 else x.nanmean(dim=1)
        x = torch.nan_to_num(x,0)
        out = self.encoder(x)
        out = self.act_1(out)
        out = self.decoder(out)
        out = self.act_2(out)
        out = (user_bias + out.T).T
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, y_mask = train_batch
        y_hat = self.forward(x)

        # set to 0 unseen by users
        y_hat *= y_mask
        y *= y_mask
        loss = self.loss_func(y_hat, y) / y_mask.sum()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, y_mask = val_batch
        y_hat = self.forward(x)

        # set to 0 unseen by users
        y_hat *= y_mask
        y *= y_mask

        loss = self.loss_func(y_hat, y_mask)
        self.log('train_loss', loss)