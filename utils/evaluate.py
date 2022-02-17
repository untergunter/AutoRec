import numpy as np
import math
import pytorch_lightning as pl
from sklearn.metrics import ndcg_score


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
