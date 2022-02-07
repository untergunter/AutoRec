from get_data import download_2_data_sets,ratings_to_train_test
from models.AutoRecBase import AutoRecBase
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


if __name__ == '__main__':
    download_2_data_sets()

    train_loader, val_loader = ratings_to_train_test(1,0, 1,10)

    model = AutoRecBase(3706,
                        25,
                        nn.PReLU,
                        nn.PReLU,
                        nn.MSELoss,
                        )
    # training
    trainer = pl.Trainer(gpus=0,max_epochs=10)
    trainer.fit(model,train_loader, val_loader)