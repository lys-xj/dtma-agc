# GNN.py
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        if adj.is_sparse:
            adj = adj.to_dense()
        output = torch.mm(adj, support)
        if active:
            output = F.leaky_relu(output, negative_slope=0.2)
        return output


class DynamicTopologyOptimizer(Module):

    def __init__(self, feature_dim, temperature=0.5):
        super(DynamicTopologyOptimizer, self).__init__()
        self.temperature = temperature
        self.alpha = Parameter(torch.tensor(0.5))
        self.feature_dim = feature_dim

    def forward(self, features, original_adj):
        n_nodes = features.shape[0]

        features_norm = F.normalize(features, p=2, dim=1)
        semantic_sim = torch.mm(features_norm, features_norm.t())

        dynamic_adj = F.softmax(semantic_sim / self.temperature, dim=1)

        if original_adj.is_sparse:
            original_adj_dense = original_adj.to_dense()
        else:
            original_adj_dense = original_adj

        alpha = torch.sigmoid(self.alpha)
        optimized_adj = alpha * original_adj_dense + (1 - alpha) * dynamic_adj

        return optimized_adj


