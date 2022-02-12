import subprocess
import os
from shutil import rmtree
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from scripts.get_data import save_obj, load_obj
from itertools import cycle

def get_2_other_datasets():
    if os.path.isfile('flixster.pkl') and os.path.isfile('douban.pkl'):
        return
    clone()
    save_data()
    rmtree('mg-gat')


def clone():
    clone_commend = 'git clone https://github.com/zuirod/mg-gat.git'.split(' ')
    subprocess.call(clone_commend)


def save_to_pickle(path: str):
    df = pd.read_csv(path)
    df = df[['user_id', 'item_id', 'rating']]
    groups_size = df.groupby('user_id').size().reset_index()
    groups_size.columns = ['user_id', 'size']
    groups_to_keep = groups_size[groups_size['size']>19]['user_id']
    df = df[df['user_id'].isin(groups_to_keep)]
    df.sort_values('user_id', inplace=True)
    ten_partitions = cycle([i for i in range(9)])
    df['partition'] = [next(ten_partitions) for count in range(len(df))]
    df.columns = ['user_id', 'item_id', 'rating', 'partition']
    name = path.split(os.sep)[-2].lower()
    save_obj(df, name)


def save_data():
    dataset_paths = ['mg-gat/data/datasets/Flixster/data.csv',
                     'mg-gat/data/datasets/Douban/data.csv']
    for data_path in dataset_paths:
        save_to_pickle(data_path)


def secondary_to_train_test(dataset_name,
                            validation_partition,
                            train_partition,
                            batch_size,
                            unseen_na_to: int = 3):
    assert dataset_name in {'flixster', 'douban'}, 'dataset  name must be flixster or douban'
    ratings = load_obj(f'{dataset_name}.pkl')

    # making sure each batch has all the movie id's
    pivot_normalizer = pd.DataFrame(
        {'item_id': ratings['item_id'].unique()})  # columns: 'user_id','item_id','rating','is_train'
    pivot_normalizer['user_id'] = -1
    pivot_normalizer['rating'] = 1
    pivot_normalizer['partition'] = 11

    pivot_normalizer = pd.DataFrame({'item_id': ratings['item_id'].unique()})
    pivot_normalizer['user_id'] = -1
    pivot_normalizer['rating'] = 1
    pivot_normalizer['partition'] = 11

    train_x = ratings[~ratings['partition'].isin({validation_partition, train_partition})]
    validation_x = ratings[~ratings['partition'].isin({validation_partition})]

    majority_train = pd.pivot_table(pd.concat([train_x, pivot_normalizer]),
                                    values='rating',
                                    index=['user_id'],
                                    columns=['item_id'])
    majority_train = majority_train[majority_train.index != -1]

    # y train is also x_test
    minority_1 = pd.pivot_table(pd.concat([validation_x, pivot_normalizer]),
                                values='rating',
                                index=['user_id'],
                                columns=['item_id'])
    minority_1 = minority_1[minority_1.index != -1]
    minority_1_na_mask = ~minority_1.isna()

    minority_2 = pd.pivot_table(ratings,
                                values='rating',
                                index=['user_id'],
                                columns=['item_id'])
    minority_2_na_mask = ~minority_2.isna()

    if unseen_na_to:
        majority_train.fillna(unseen_na_to, inplace=True)
        minority_1.fillna(unseen_na_to, inplace=True)
        minority_2.fillna(unseen_na_to, inplace=True)

    X_train_tensor = torch.tensor(majority_train.values.astype(np.float32))
    Y_train_tensor = torch.tensor(minority_1.values.astype(np.float32))
    Y_test_tensor = torch.tensor(minority_2.values.astype(np.float32))

    Y_train_is_not_na = torch.tensor(minority_1_na_mask.values.astype(np.float32))
    Y_test_is_not_na = torch.tensor(minority_2_na_mask.values.astype(np.float32))

    train_tensor = data_utils.TensorDataset(X_train_tensor, Y_train_tensor, Y_train_is_not_na)
    test_tensor = data_utils.TensorDataset(Y_train_tensor, Y_test_tensor, Y_test_is_not_na)

    train_loader = data_utils.DataLoader(dataset=train_tensor,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)

    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=batch_size,
                                        shuffle=False)  # for results evaluation

    return train_loader, test_loader

