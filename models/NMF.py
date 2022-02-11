import torch
import torch.nn as nn
import pytorch_lightning as pl


class NMF(pl.LightningModule):

    def __init__(self, num_users=6040, num_items=3706, latent_dim_mf=8, latent_dim_mlp=8):
        super(NMF, self).__init__()

        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp

        # Embedding layers:
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=self.latent_dim_mf)
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=self.latent_dim_mlp)

        # MLP Layers:
        self.fc_layers = torch.nn.ModuleList()
        self.fc_layer0 = torch.nn.Linear(in_features=16, out_features=32)
        self.fc_layer1 = torch.nn.Linear(in_features=32, out_features=16)
        self.fc_layer2 = torch.nn.Linear(in_features=16, out_features=8)
        self.logits = torch.nn.Linear(in_features=16, out_features=1)

    def forward(self, user_vec, item_vec):
        # Embed:
        mf_user_vec = self.embedding_user_mf(user_vec)
        mf_item_vec = self.embedding_item_mf(item_vec)
        mlp_user_vec = self.embedding_user_mlp(user_vec)
        mlp_item_vec = self.embedding_item_mlp(item_vec)

        # Flatten:
        mf_user_vec = mf_user_vec.view(-1, self.latent_dim_mf)
        mf_item_vec = mf_item_vec.view(-1, self.latent_dim_mf)
        mlp_user_vec = mlp_user_vec.view(-1, self.latent_dim_mlp)
        mlp_item_vec = mlp_item_vec.view(-1, self.latent_dim_mlp)

        # MLP:
        mlp_vec = torch.cat([mlp_user_vec, mlp_item_vec], 1)
        mlp_vec = self.fc_layer0(mlp_vec)
        mlp_vec = self.fc_layer1(mlp_vec)
        mlp_vec = self.fc_layer2(mlp_vec)

        # MF:
        mf_vec = torch.mul(mf_user_vec, mf_item_vec)
        x = torch.cat([mlp_vec, mf_vec], 1)
        x = self.logits(x)
        return x

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