import numpy as np
import pandas as pd
import requests
import zipfile
import os
import pickle
from shutil import rmtree
from itertools import cycle
import torch
import torch.utils.data as data_utils


def download_ratings_1_10_m():
    for data_set_size in (1, 10):
        url = f'http://files.grouplens.org/datasets/movielens/ml-{data_set_size}m.zip'
        zip_name = f'ml-{data_set_size}m.zip'

        r = requests.get(url, allow_redirects=True)
        open(zip_name, 'wb').write(r.content)

        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall()

        os.remove(zip_name)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def keep_ratings_add_cv_partition():
    for folder in ('ml-1m', 'ml-10M100K'):
        ratings = pd.read_table(
            f'{folder}/ratings.dat',
            header=None,
            delimiter='::',
            usecols=[0, 1, 2],
            names=['user_id', 'movie_id', 'rating'])
        ten_partitions = cycle([i for i in range(9)])  # each user appears at least twice in each partition
        ratings.sort_values('user_id', inplace=True)
        ratings['partition'] = [next(ten_partitions) for count in range(len(ratings))]
        name = folder.replace('100K', '').lower()
        save_obj(ratings, name)
        rmtree(folder)


def ratings_to_train_test(dataset_size,
                          validation_partition,
                          train_partition,
                          batch_size):
    assert dataset_size in {1, 10}, 'datasets are ml-1m and ml-10m, size must be 1 or 10'
    assert validation_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert train_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert validation_partition != train_partition
    ratings = load_obj(f'ml-{dataset_size}m.pkl')

    pivot_nrmlizer = pd.DataFrame({'movie_id': ratings['movie_id'].unique()})
    pivot_nrmlizer['user_id'] = -1
    pivot_nrmlizer['rating'] = 1
    pivot_nrmlizer['partition'] = 11

    train_x = ratings[~ratings['partition'].isin({validation_partition, train_partition})]
    validation_x = ratings[~ratings['partition'].isin({validation_partition})]

    X_train = pd.pivot_table(pd.concat([train_x, pivot_nrmlizer]),
                             values='rating',
                             index=['user_id'],
                             columns=['movie_id'])
    X_train = X_train[X_train.index != -1]
    X_train.fillna(3, inplace=True)

    # y train is also x_test
    Y_train = pd.pivot_table(pd.concat([validation_x, pivot_nrmlizer]),
                             values='rating',
                             index=['user_id'],
                             columns=['movie_id'])
    Y_train = Y_train[Y_train.index != -1]
    Y_train.fillna(3, inplace=True)

    Y_test = pd.pivot_table(ratings,
                            values='rating',
                            index=['user_id'],
                            columns=['movie_id'])
    Y_test.fillna(3, inplace=True)

    batch_index = torch.tensor(X_train.index.values)
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    Y_train_tensor = torch.tensor(Y_train.values.astype(np.float32))
    Y_test_tensor = torch.tensor(Y_test.values.astype(np.float32))

    train_tensor = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
    test_tensor = data_utils.TensorDataset(Y_train_tensor, Y_test_tensor)

    train_loader = data_utils.DataLoader(dataset=train_tensor,
                                         batch_size=batch_size,
                                         shuffle=False)
    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=batch_size,
                                        shuffle=False)  # for results debugging

    return train_loader, test_loader


def download_2_data_sets():
    if os.path.isfile('ml-1m.pkl') and os.path.isfile('ml-10m.pkl'):
        return
    download_ratings_1_10_m()
    keep_ratings_add_cv_partition()

if __name__ == '__main__':
    download_2_data_sets()
    train, test = ratings_to_train_test(1,
                                        0,
                                        1,
                                        10)