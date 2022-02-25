import numpy as np
import math
import pytorch_lightning as pl
import torch
from sklearn.metrics import ndcg_score,mean_squared_error


def evaluate_model(model: pl.LightningModule, test_loader, K=10):
    """
    Evaluate the performance (Hit_Ratio, NDCG, MRR) of top-K recommendation
    Return: score of each test rating.
    """
    print(model)
    hits, ndcgs, mrrs = [],[],[]

    for X, Y, mask in test_loader:
        if type(model).__name__ == "VarAutoRec":
            _, _, _, predictions = model(X)
            pre = predictions.squeeze().detach()
            predictions = np.argsort(pre, axis=1)
        else:
            pre = model(X).squeeze().detach()
            predictions = np.argsort(pre, axis=1)
        sorted_ground = np.argsort(X*mask,axis=1)
        for i in range(predictions.shape[0]):
            (hr, ndcg, mrr) = eval_one_rating(predictions[i, :K],
                                              np.where(mask[i,:]==1)[0],
                                              sorted_ground[i,:],
                                              pre[i,:],
                                              Y*mask[i:],
                                              K
                                              )

            hits.append(hr)
            ndcgs.append(ndcg)
            mrrs.append(mrr)

    return hits, ndcgs, mrrs


def eval_one_rating(predictions,mask,sg,raw_pred,masked_true,k):
    # Evaluate top rank list
    hr = get_hit_ratio(predictions,mask)
    ndcg = get_ndcg(masked_true,raw_pred,k)
    mrr = get_mrr(predictions,sg)
    return hr, ndcg, mrr


def get_hit_ratio(ranklist,mask):
    hr = sum(i for i in mask if i in ranklist)/ max(len(mask),len(ranklist))
    return hr


def get_ndcg(masked_true,raw_pred,k):
    ndcg = ndcg_score(masked_true, raw_pred,k=k)
    return ndcg


def get_mrr(ranklist,sg):
    for i in range(len(ranklist)):
        if ranklist[i] in sg:
            return 1/(i+1)
    return 0

def rmse(true,pred,n):
    error = true-pred
    square_error = error**2
    mse = np.sum(square_error)/n
    rmse = np.sqrt(mse)
    return rmse

def get_rmse(true,pred):
    rmse = mean_squared_error(true,pred,squared=False)
    return rmse

def evaluate_model_rmse(model: pl.LightningModule, test_loader, K=10):
    """
    Evaluate the performance (Hit_Ratio, NDCG, MRR) of top-K recommendation
    Return: score of each test rating.
    """
    print(model)
    y_true,y_mask,y_pred,only_in_pred_mask = [],[],[],[]

    for X, Y, mask in test_loader:

        mask_only_pred = (torch.nan_to_num(Y-X,0)!=0) * 1
        y_true.append(Y)
        y_mask.append(mask)
        only_in_pred_mask.append(mask_only_pred)
        if type(model).__name__ == "VarAutoRec":
            _, _, _, predictions = model(X)
            pre = predictions.squeeze().detach()
            predictions = np.argsort(pre, axis=1)
        else:
            pre = model(X).squeeze().detach()
            predictions = np.argsort(pre, axis=1)
        y_pred.append(predictions)

    y_true = np.array(torch.cat(y_true))
    y_mask = np.array(torch.cat(y_mask))*1
    y_pred = np.array(torch.cat(y_pred))
    only_in_pred_mask = np.array(torch.cat(only_in_pred_mask))

    y_true_all = y_true[y_mask==1]
    y_pred_all = y_pred[y_mask==1]
    n_views = y_mask.sum()
    all_rmse_score = get_rmse(y_true_all,y_pred_all)

    y_true = y_true[only_in_pred_mask==1]
    y_pred = y_pred[only_in_pred_mask==1]
    n_views = only_in_pred_mask.sum()
    only_pred_rmse_score = get_rmse(y_true,y_pred)

    return all_rmse_score,only_pred_rmse_score