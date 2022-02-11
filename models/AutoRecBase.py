import torch
import torch.nn as nn
import pytorch_lightning as pl


class AutoRecBase(pl.LightningModule):
    def __init__(self,
                 number_of_items: int,
                 hidden_size: int,
                 activation_function_1,
                 activation_function_2,
                 loss,
                 lr=0.001):
        super(AutoRecBase, self).__init__()

        self.encoder = nn.Linear(number_of_items, hidden_size)
        self.act_1 = activation_function_1()
        self.decoder = nn.Linear(hidden_size, number_of_items)
        self.act_2 = activation_function_2()
        self.loss_func = loss()
        self.lr = lr

    def forward(self, x):
        out = self.encoder(x)
        out = self.act_1(out)
        out = self.decoder(out)
        out = self.act_2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, y_mask = train_batch
        y_hat = self.forward(x)

        # set to 0 unseen by users
        y_hat *= y_mask
        y *= y_mask
        loss = self.loss_func(y_hat, y_mask) / y_mask.sum()
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