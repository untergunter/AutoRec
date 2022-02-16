import torch
import torch.nn as nn
import pytorch_lightning as pl


class MF(pl.LightningModule):

    def __init__(
            self,
            number_of_items: int,
            num_of_users: int,
            hidden_size: int,
            activation_function_1,
            activation_function_2,
            loss,
            λ=0.01,
            lr=0.001
    ):
        super(MF, self).__init__()
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=num_of_users, embedding_dim=hidden_size)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=number_of_items, embedding_dim=hidden_size)
        self.l_0 = nn.Linear(20, 1)
        self.loss_func = nn.L1Loss()
        self.lr = lr

    def forward(self, user_vec, item_vec):
        user_vec = self.embedding_user_mf(user_vec)
        item_vec = self.embedding_item_mf(item_vec)
        user_vec = user_vec.view(-1, 20)
        item_vec = item_vec.view(-1, 20)
        x = torch.mul(user_vec, item_vec)
        x = self.l_0(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, r = train_batch
        r_hat = self.forward(x, y)
        loss = self.loss_func(r_hat, r)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, r = val_batch
        r_hat = self.forward(x, y)
        loss = self.loss_func(r_hat, r)
        self.log('val_loss', loss)
