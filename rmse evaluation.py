import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# Our Stuff:
from models.AutoRecBase import AutoRecBase
from models.VarAutoRec import VarAutoRec
from models.MF import MF
from models.UserBiasAE import UserBiasAE
from glob import glob

from scripts.get_data import download_2_data_sets, ratings_to_train_test, ratings_to_train_test_u
from scripts.get_2_other_data import get_2_other_datasets, secondary_to_train_test, secondary_to_train_test_u
from utils.evaluate import evaluate_model_rmse
from utils.loading_utils import load_model, save_model

import torch
from torch import nn
import pytorch_lightning as pl

is_default_dataset = True

models = [
    AutoRecBase,
    VarAutoRec,
    UserBiasAE
]
lrs = [0.001,0.005,0.01]
activations = [nn.PReLU, nn.Sigmoid]
latent_dims = [10, 80, 300]
lambdas = [0.01, 0.1, 1, 100]


models_from_string = {'AutoRecBase':AutoRecBase,
           'MF':MF,
           'UserBiasAE':UserBiasAE,
           'VarAutoRec':VarAutoRec}

def attr_from_path(path):
    class_name,activation,hidden_size,lr,λ,data_set_name = path.split(os.sep)[-1].replace('_model_dict.ckpt','').split('_')
    class_to_take = models_from_string[class_name]
    return class_to_take,activation,hidden_size,lr,λ,data_set_name

if __name__ == '__main__':


    if is_default_dataset:
        download_2_data_sets()
    else:
        get_2_other_datasets()

    models_eval_dict = {}
    Ks = [10, 20, 50, 100]
    i = 0

    if is_default_dataset:
        train_loader, val_loader = ratings_to_train_test(1, 0, 1, 10)
        mf_train_loader, mf_val_loader = ratings_to_train_test_u(dataset_size=1,
                                                                 validation_partition=0,
                                                                 train_partition=1,
                                                                 batch_size=10)
    else:
        train_loader, val_loader = secondary_to_train_test('douban', 0, 1, 10)
        mf_train_loader, mf_val_loader = secondary_to_train_test_u('douban',
                                                                   validation_partition=0,
                                                                   train_partition=1,
                                                                   batch_size=10)
    obs = [path for path in
           glob('/home/ido/data/idc/Recommender Systems/AutoRec/obj/*.ckpt')
           if 'True' in path and not 'MF' in path ]

    cls, act, hs, lrs, lambdas, all_rmse, val_rmse, ks = [], [], [], [], [], [], [], []
    print(len(obs))
    for path in tqdm(obs):
        class_to_take, activation, hidden_size, lr, λ, data_set_name = attr_from_path(path)
        print(class_to_take)
        continue
        try:
            model = class_to_take.load_from_checkpoint(path)
            model.eval()
            for K in Ks:
                all_rmse_score, only_pred_rmse_score = evaluate_model_rmse(model,
                                                                           test_loader=val_loader,
                                                                           K=K)
                cls.append(type(model).__name__)
                act.append(activation)
                hs.append(hidden_size)
                lrs.append(lr)
                lambdas.append(λ)
                all_rmse.append(all_rmse_score)
                val_rmse.append(only_pred_rmse_score)
                ks.append(K)
        except Exception:
            print(class_to_take)

    eval_df = pd.DataFrame({'model_name':cls,
                            'activation':act,
                            'hidden_size':hs,
                            'learning_rate':lrs,
                            'lambda':lambdas,
                            'all_rmse':all_rmse,
                            'test_rmse':val_rmse,
                            'topk': ks
                            })
    eval_df['dataset'] = 'douban'
    eval_df.to_csv("eval_df_ml1_rmse.csv", sep='\t')

    eval_df.head()