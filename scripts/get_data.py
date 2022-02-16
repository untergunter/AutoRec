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
        ratings.sort_values('user_id', inplace=True)
        ten_partitions = cycle([i for i in range(9)])  # each user appears at least twice in each partition
        ratings['partition'] = [next(ten_partitions) for count in range(len(ratings))]
        name = folder.replace('100K', '').lower()
        save_obj(ratings, name)
        rmtree(folder)


def ratings_to_train_test(dataset_size,
                          validation_partition,
                          train_partition,
                          batch_size,
                          unseen_na_to: int = 3):
    assert dataset_size in {1, 10}, 'datasets are ml-1m and ml-10m, size must be 1 or 10'
    assert validation_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert train_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert validation_partition != train_partition
    ratings = load_obj(f'ml-{dataset_size}m.pkl')

    # making sure each batch has all the movie id's
    pivot_normalizer = pd.DataFrame({'movie_id': ratings['movie_id'].unique()})
    pivot_normalizer['user_id'] = -1
    pivot_normalizer['rating'] = 1
    pivot_normalizer['partition'] = 11

    train_x = ratings[~ratings['partition'].isin({validation_partition, train_partition})]
    validation_x = ratings[~ratings['partition'].isin({validation_partition})]

    majority_train = pd.pivot_table(pd.concat([train_x, pivot_normalizer]),
                                    values='rating',
                                    index=['user_id'],
                                    columns=['movie_id'])
    majority_train = majority_train[majority_train.index != -1]

    # y train is also x_test
    minority_1 = pd.pivot_table(pd.concat([validation_x, pivot_normalizer]),
                                values='rating',
                                index=['user_id'],
                                columns=['movie_id'])
    minority_1 = minority_1[minority_1.index != -1]
    minority_1_na_mask = ~minority_1.isna()

    minority_2 = pd.pivot_table(ratings,
                                values='rating',
                                index=['user_id'],
                                columns=['movie_id'])
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
                                         num_workers=16)

    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=16)  # for results evaluation

    return train_loader, test_loader


def ratings_to_train_test_u(dataset_size,
                          validation_partition,
                          train_partition,
                          batch_size):
    assert dataset_size in {1, 10}, 'datasets are ml-1m and ml-10m, size must be 1 or 10'
    assert validation_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert train_partition in set(i for i in range(10)), 'using 10 cross validations'
    assert validation_partition != train_partition
    ratings = load_obj(f'ml-{dataset_size}m.pkl')

    train_x = ratings[~ratings['partition'].isin({validation_partition, train_partition})]
    validation_x = ratings[~ratings['partition'].isin({validation_partition})]

    train_user = torch.tensor(train_x['user_id'].values)
    train_rating = torch.tensor(train_x['rating'].values)
    train_item = torch.tensor(train_x['movie_id'].values)

    train_tensor = data_utils.TensorDataset(train_user, train_item, train_rating)
    train_loader = data_utils.DataLoader(dataset=train_tensor,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=16)

    test_user = torch.tensor(validation_x['user_id'].values)
    test_rating = torch.tensor(validation_x['rating'].values)
    test_item = torch.tensor(validation_x['movie_id'].values)

    test_tensor = data_utils.TensorDataset(test_user, test_item, test_rating)
    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=16)

    return train_loader, test_loader


def download_2_data_sets():
    if os.path.isfile('ml-1m.pkl') and os.path.isfile('ml-10m.pkl'):
        return
    download_ratings_1_10_m()
    keep_ratings_add_cv_partition()
