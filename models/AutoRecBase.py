import torch
import torch.nn as nn
import pytorch_lightning as pl

class AutoRecBase(pl.LightningModule):
    def __init__(self,
                 number_of_items: int,
                 hidden_size: int,
                 activation_function_1,
                 activation_function_2,
                 loss):
        super(AutoRecBase, self).__init__()

        self.encoder = nn.Linear(number_of_items, hidden_size)
        self.act_1 = activation_function_1()
        self.decoder = nn.Linear(hidden_size, number_of_items)
        self.act_2 = activation_function_2()
        self.loss_func = loss()

    def forward(self, x):
        out = self.encoder(x)
        out = self.act_1(out)
        out = self.decoder(out)
        out = self.act_2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.forward(x)
        loss = self.loss_func(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.forward(x)
        loss = self.loss_func(x_hat, x)
        self.log('val_loss', loss)