import numpy as np


def normalize_l1(features):
    row_norms = np.abs(features).sum(axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-8
    return features / row_norms


def pairwise_euclidean_dist(features):
    features_sq = np.sum(features ** 2, axis=1, keepdims=True)
    dist_sq = features_sq + features_sq.T - 2 * np.dot(features, features.T)
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)
def construct_graph(features, label, method='ncos', dynamic_k=True, dataset_name='dblp'):
# def construct_graph(features, label, method='ncos', dynamic_k=True, dataset_name='hhar'):
# def construct_graph(features, label, method='ncos', dynamic_k=True, dataset_name='reut'):
# def construct_graph(features, label, method='ncos', dynamic_k=True, dataset_name='acm'):
    import os
    if not os.path.exists('graph'):
        os.makedirs('graph')

    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pairwise_euclidean_dist(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize_l1(features)
        dist = np.dot(features, features.T)

    if dynamic_k:
        avg_sim = np.mean(dist)
        std_sim = np.std(dist)
        if avg_sim > 0.7:
            topk = max(5, min(10, int(8 - (avg_sim - 0.7) / 0.05)))
        elif avg_sim > 0.4:
            topk = max(10, min(15, int(12 - (avg_sim - 0.4) / 0.03)))
        else:
            topk = max(15, min(20, int(18 - (avg_sim - 0.2) / 0.02)))
        print(f"Dynamic K value adjusted to: {topk}")

        k_value_path = f'graph/{dataset_name}_k_value.txt'
        with open(k_value_path, 'w') as f:
            f.write(str(topk))
        print(f"K value saved to: {k_value_path}")
    else:
        topk = 10

    fname = f'graph/{dataset_name}{topk}_graph.txt'
    inds = []
    for i in range(dist.shape[0]):
        threshold = np.mean(dist[i, :]) - 0.1 * std_sim
        candidates = np.where(dist[i, :] > threshold)[0]
        if len(candidates) > topk + 1:
            ind = np.argpartition(dist[i, candidates], -(topk + 1))[-(topk + 1):]
            inds.append(candidates[ind])
        else:
            ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
            inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                weight = dist[i, vv]
                if label[vv] != label[i]:
                    counter += 1
                f.write(f'{i} {vv} {weight:.4f}\n')
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))
    print(f"Graph file generated: {fname}")

    return topk

#
# # Data loading
# hhar = np.loadtxt('data/hhar.txt', dtype=float)
# label = np.loadtxt('data/hhar_label.txt', dtype=int)
#
# # Generate graph and get dynamic k value
# dynamic_k_value = construct_graph(hhar, label, 'ncos', dynamic_k=True, dataset_name='hhar')
# print(f"Final dynamic K value: {dynamic_k_value}")


# Data loading
# acm = np.loadtxt('data/acm.txt', dtype=float)
# label = np.loadtxt('data/acm_label.txt', dtype=int)
#
# # Generate graph and get dynamic k value
# dynamic_k_value = construct_graph(acm, label, 'ncos', dynamic_k=True, dataset_name='acm')
# print(f"Final dynamic K value: {dynamic_k_value}")
# reut = np.loadtxt('data/reut.txt', dtype=float)
# label = np.loadtxt('data/reut_label.txt', dtype=int)
# # Generate graph file + dynamic k value file for reut
# dynamic_k_value = construct_graph(reut, label, 'ncos', dynamic_k=True, dataset_name='reut')
# print(f"Final dynamic K value: {dynamic_k_value}")


dblp= np.loadtxt('data/dblp.txt', dtype=float)
label = np.loadtxt('data/dblp_label.txt', dtype=int)
# Generate graph file + dynamic k value file for dblp
dynamic_k_value = construct_graph(dblp, label, 'ncos', dynamic_k=True, dataset_name='dblp')
print(f"Final dynamic K value: {dynamic_k_value}")