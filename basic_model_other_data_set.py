from scripts.get_2_other_data import get_2_other_datasets,secondary_to_train_test
from models.AutoRecBase import AutoRecBase
import torch.nn as nn
import torch.cuda
import pytorch_lightning as pl

if __name__ == '__main__':
    VALIDATION_PARTITION = 0
    gpu = 1 if torch.cuda.is_available() else 0
    get_2_other_datasets()
    model = AutoRecBase(2876,
                        25,
                        nn.PReLU,
                        nn.PReLU,
                        nn.MSELoss,
                        )
    # training
    trainer = pl.Trainer(gpus=gpu,max_epochs=10)
    for train_partition in range(10):
        if train_partition != VALIDATION_PARTITION:
            train_loader, val_loader = secondary_to_train_test('flixster', VALIDATION_PARTITION, train_partition, 10)

            trainer.fit(model,train_loader, val_loader)