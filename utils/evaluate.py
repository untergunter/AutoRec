import numpy as np
import math
import pytorch_lightning as pl


def evaluate_model(model: pl.LightningModule, test_loader, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG, MRR) of top-K recommendation
    Return: score of each test rating.
    """
    print(model)
    hits, ndcgs, mrrs = [],[],[]

    for user, item, rate in test_loader:
        predictions = np.argsort(model(user, item).squeeze().detach())

        (hr, ndcg, mrr) = eval_one_rating(predictions[:K])

        hits.append(hr)
        ndcgs.append(ndcg)
        mrrs.append(mrr)

    return hits, ndcgs, mrrs


def eval_one_rating(predictions):
    # Evaluate top rank list
    hr = get_hit_ratio(predictions)
    ndcg = get_ndcg(predictions)
    mrr = get_mrr(predictions)
    return hr, ndcg, mrr


def get_hit_ratio(ranklist):
    return 0 in ranklist


def get_ndcg(ranklist):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == 0:
            return math.log(2) / math.log(i+2)
    return 0


def get_mrr(ranklist):
    for i in range(len(ranklist)):
        if ranklist[i] == 0:
            return 1/(i + 1)
    return 0
