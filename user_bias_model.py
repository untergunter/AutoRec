from scripts.get_data import download_2_data_sets,ratings_to_train_test
from models.UserBiasAE import UserBiasAE
import torch.nn as nn
import torch.cuda
import pytorch_lightning as pl


if __name__ == '__main__':

    VALIDATION_PARTITION = 0
    gpu = 1 if torch.cuda.is_available() else 0
    download_2_data_sets()
    model = UserBiasAE(3706,
                       25,
                       nn.PReLU,
                       nn.PReLU,
                       nn.MSELoss,
                       )
    # training

    for train_partition in range(10):
        if train_partition != VALIDATION_PARTITION:
            train_loader, val_loader = ratings_to_train_test(1, VALIDATION_PARTITION, train_partition, 256,None)
            trainer = pl.Trainer(gpus=gpu, max_epochs=10)
            trainer.fit(model,train_loader, val_loader)