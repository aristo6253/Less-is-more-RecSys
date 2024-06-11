import torch
import numpy as np

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def metrics(uids, predictions, topk, test_labels, test_ratings):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            # old code:
            # idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            # new code:
            ideal_scores = sorted(test_ratings[uid], reverse=True)
            if len(ideal_scores) > topk:
              ideal_scores = ideal_scores[:topk]
            idcg = np.sum([((2 ** ideal_scores[loc] - 1) / np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    # Old code:
                    # dcg = dcg + np.reciprocal(np.log2(loc+2))
                    # New code:
                    loc_rating = label.index(item)
                    dcg = dcg + ((2 ** test_ratings[uid][loc_rating] - 1) / np.log2(loc + 2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

