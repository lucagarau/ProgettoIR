import torch
import numpy as np
import math
from collections import Counter
from tqdm import tqdm


def get_hit_k(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    return np.round(hit / pred_rank.shape[0], decimals=5)

def get_hit_k_gt5(ratings, pred_rank, k):

    item_counts = Counter([item for _, item in ratings])
    selected_items = set([item for item, count in item_counts.items() if count >= 5])

    # Indici delle interazioni con item frequenti
    filtered_idx = [i for i, (_, item) in enumerate(ratings) if item in selected_items]

    if not filtered_idx:
        return 0.0

    pred_rank_k = pred_rank[filtered_idx, :k]
    hit = np.count_nonzero(pred_rank_k == 0)

    return np.round(hit / pred_rank.shape[0], decimals=5)

def get_hit_k_lt5(ratings, pred_rank, k):

    item_counts = Counter([item for _, item in ratings])
    selected_items = set([item for item, count in item_counts.items() if count < 5])

    # Indici delle interazioni con item frequenti
    filtered_idx = [i for i, (_, item) in enumerate(ratings) if item in selected_items]

    if not filtered_idx:
        return 0.0

    pred_rank_k = pred_rank[filtered_idx, :k]
    hit = np.count_nonzero(pred_rank_k == 0)

    return np.round(hit / pred_rank.shape[0], decimals=5)


def get_ndcg_k(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    return np.round(np.mean(ndcgs), decimals=5)

def get_ndcg_k_split(ratings, pred_rank, k):
    # Conta le occorrenze degli item
    item_counts = Counter([item for _, item in ratings])

    # Popolari: >5 occorrenze, Non popolari: ==5
    popular_items = set([item for item, count in item_counts.items() if count >= 5])
    non_popular_items = set([item for item, count in item_counts.items() if count < 5])

    # Liste per ndcg popolari e non popolari
    ndcg_popular = []
    ndcg_non_popular = []

    for i, (_, true_item) in enumerate(ratings):
        # Trova la posizione in cui si trova l’item corretto (rappresentato da 0)
        found = False
        for j in range(k):
            if pred_rank[i][j] == 0:
                dcg = math.log(2) / math.log(j + 2)  # posizione j → log(j+2)
                found = True
                break
        if not found:
            dcg = 0.0

        # Assegna all’insieme corretto
        if true_item in popular_items:
            ndcg_popular.append(dcg)
        elif true_item in non_popular_items:
            ndcg_non_popular.append(dcg)

    # Calcolo medie (evitiamo divisione per 0)
    ndcg_pop = np.round(np.mean(ndcg_popular), 5) if ndcg_popular else 0.0
    ndcg_nonpop = np.round(np.mean(ndcg_non_popular), 5) if ndcg_non_popular else 0.0

    return ndcg_pop, ndcg_nonpop


def get_mrr_k(pred_rank, k):
    mrrs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                mrrs[user] = 1.0 / (j + 1)
                break
    return np.round(np.mean(mrrs), decimals=5)



def get_mrp_k_split(ratings, pred_rank, k):
    item_counts = Counter([item for _, item in ratings])
    popular_items = set([item for item, count in item_counts.items() if count >= 5])
    non_popular_items = set([item for item, count in item_counts.items() if count < 5])

    mrp_popular = []
    mrp_non_popular = []

    for i, (_, true_item) in enumerate(ratings):
        reciprocal = 0.0
        for j in range(k):
            if pred_rank[i][j] == 0:
                reciprocal = 1.0 / (j + 1)
                break

        if true_item in popular_items:
            mrp_popular.append(reciprocal)
        elif true_item in non_popular_items:
            mrp_non_popular.append(reciprocal)

    mrp_pop = np.round(np.mean(mrp_popular), 5) if mrp_popular else 0.0
    mrp_nonpop = np.round(np.mean(mrp_non_popular), 5) if mrp_non_popular else 0.0

    return mrp_pop, mrp_nonpop




def evaluate(model, ratings, negatives, device, k_list, type_m="group"):
    hits_K, ndcgs_K, mrrs_K = [], [], []

    topK_rank_array = np.zeros((len(ratings), max(k_list)))

    for idx in range(len(ratings)):
        user_test, item_test = [], []

        rating = ratings[idx]
        # candidate 0 is pos_item
        items = [rating[1]]
        items.extend(negatives[idx])

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

        users_var = torch.LongTensor(np.array(user_test)).view(-1).to(device)
        items_var = torch.LongTensor(np.array(item_test)).view(-1).to(device)

        predictions = model(users_var, items_var, type_m).squeeze()
        pred_score = predictions.data.cpu().numpy().reshape(1, -1)
        pred_rank = np.argsort(pred_score * -1, axis=1)

        topK_rank_array[idx, :] = pred_rank[0, : max(k_list)]

    for k in k_list:
        hits_K.append(get_hit_k(topK_rank_array, k))
        ndcgs_K.append(get_ndcg_k(topK_rank_array, k))
        mrrs_K.append(get_mrr_k(topK_rank_array, k))

    return hits_K, ndcgs_K, mrrs_K

def get_mrr_k_split(ratings, pred_rank, k):
    item_counts = Counter([item for _, item in ratings])
    popular_items = set([item for item, count in item_counts.items() if count >= 5])
    non_popular_items = set([item for item, count in item_counts.items() if count < 5])

    mrr_popular, mrr_non_popular = [], []

    for i, (_, true_item) in enumerate(ratings):
        reciprocal = 0.0
        for j in range(k):
            if pred_rank[i][j] == 0:
                reciprocal = 1.0 / (j + 1)
                break

        if true_item in popular_items:
            mrr_popular.append(reciprocal)
        elif true_item in non_popular_items:
            mrr_non_popular.append(reciprocal)

    mrr_pop = np.round(np.mean(mrr_popular), 5) if mrr_popular else 0.0
    mrr_nonpop = np.round(np.mean(mrr_non_popular), 5) if mrr_non_popular else 0.0

    return mrr_pop, mrr_nonpop


def evaluate2(model, ratings, negatives, device, k_list, type_m="group"):
    hits_K, ndcgs_K, mrrs_K = [], [], []
    hits_K_gt5, hits_K_lt5 = [], []
    ndcg_pop, ndcg_npop = [], []
    mrr_pop_list, mrr_npop_list = [], []
    mrp_pop_list, mrp_npop_list = [], []
    mrp_total_list = []

    topK_rank_array = np.zeros((len(ratings), max(k_list)))

    for idx in tqdm(range(len(ratings)), desc="Processing ratings"):
        user_test, item_test = [], []

        rating = ratings[idx]
        items = [rating[1]]
        items.extend(negatives[idx])

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

        users_var = torch.LongTensor(np.array(user_test)).view(-1).to(device)
        items_var = torch.LongTensor(np.array(item_test)).view(-1).to(device)

        predictions = model(users_var, items_var, type_m).squeeze()
        pred_score = predictions.data.cpu().numpy().reshape(1, -1)
        pred_rank = np.argsort(pred_score * -1, axis=1)

        topK_rank_array[idx, :] = pred_rank[0, : max(k_list)]

    for k in tqdm(k_list, desc="Processing metrics"):
        hits_K.append(get_hit_k(topK_rank_array, k))
        ndcgs_K.append(get_ndcg_k(topK_rank_array, k))
        mrrs_K.append(get_mrr_k(topK_rank_array, k))

        hits_K_gt5.append(get_hit_k_gt5(ratings, topK_rank_array, k))
        hits_K_lt5.append(get_hit_k_lt5(ratings, topK_rank_array, k))

        pop_ndcg, npop_ndcg = get_ndcg_k_split(ratings, topK_rank_array, k)
        ndcg_pop.append(pop_ndcg)
        ndcg_npop.append(npop_ndcg)

        mrp_pop, mrp_npop = get_mrp_k_split(ratings, topK_rank_array, k)
        mrp_pop_list.append(mrp_pop)
        mrp_npop_list.append(mrp_npop)

        mrr_pop, mrr_npop = get_mrr_k_split(ratings, topK_rank_array, k)
        mrr_pop_list.append(mrr_pop)
        mrr_npop_list.append(mrr_npop)

        mrp_total = get_mrr_k(topK_rank_array, k)  # MRP = MRR on whole set
        mrp_total_list.append(mrp_total)

    return (
        hits_K, ndcgs_K, mrrs_K,
        hits_K_gt5, hits_K_lt5,
        ndcg_pop, ndcg_npop,
        mrp_pop_list, mrp_npop_list,
        mrr_pop_list, mrr_npop_list,
        mrp_total_list
    )

