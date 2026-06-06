import warnings
import os
warnings.filterwarnings("ignore", category=RuntimeWarning, message="scipy._lib.messagestream")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores")

import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
import warnings
import time
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

def load_dynamic_k_value(dataset_name):
    k_value_path = f'graph/{dataset_name}_k_value.txt'
    try:
        with open(k_value_path, 'r') as f:
            k_value = int(f.read().strip())
        print(f"load dynamic K from {k_value_path}: {k_value}")
        return k_value
    except FileNotFoundError:
        print(f"warning: dynamic K file not found {k_value_path}, use default")
        return 5

class LinearWithAct(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.linear(x))

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)
        return weight_output

class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)
        return weight_output

class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output

class DimensionAttention(nn.Module):
    def __init__(self, feat_dim, reduction_ratio=4):
        super(DimensionAttention, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = max(1, feat_dim // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hidden_dim, feat_dim)
        )

        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 5.0)

    def forward(self, x):
        attn_logits = self.mlp(x)
        attn = torch.sigmoid(attn_logits)
        out = x * attn
        return out

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        self.attention_enc = DimensionAttention(n_enc_3)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        enc_h3_attn = self.attention_enc(enc_h3)

        z = self.z_layer(enc_h3_attn)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z

class DTMA(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(DTMA, self).__init__()
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location=device), strict=False)

        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(n_z, n_clusters)

        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)
        self.mlp_scale = MLP_L(5 * n_z)

        self.fc_z1 = Linear(n_enc_1, n_z)
        self.fc_z2 = Linear(n_enc_2, n_z)
        self.fc_z3 = Linear(n_enc_3, n_z)
        self.fc_z4 = Linear(n_z, n_z)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        n_x = x.shape[0]

        z1 = self.agcn_0(x, adj)
        m1 = self.mlp1(torch.cat((tra1, z1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = m1[:, 0].reshape(n_x, 1).repeat(1, 500)
        m12 = m1[:, 1].reshape(n_x, 1).repeat(1, 500)
        z1_fused = m11 * z1 + m12 * tra1
        z1_mapped = self.fc_z1(z1_fused)

        z2 = self.agcn_1(z1_fused, adj)
        m2 = self.mlp2(torch.cat((tra2, z2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = m2[:, 0].reshape(n_x, 1).repeat(1, 500)
        m22 = m2[:, 1].reshape(n_x, 1).repeat(1, 500)
        z2_fused = m21 * z2 + m22 * tra2
        z2_mapped = self.fc_z2(z2_fused)

        z3 = self.agcn_2(z2_fused, adj)
        m3 = self.mlp3(torch.cat((tra3, z3), 1))
        m3 = F.normalize(m3, p=2)
        m31 = m3[:, 0].reshape(n_x, 1).repeat(1, 2000)
        m32 = m3[:, 1].reshape(n_x, 1).repeat(1, 2000)
        z3_fused = m31 * z3 + m32 * tra3
        z3_mapped = self.fc_z3(z3_fused)

        z4 = self.agcn_3(z3_fused, adj)
        z4_mapped = self.fc_z4(z4)

        scale_features = torch.cat((z1_mapped, z2_mapped, z3_mapped, z4_mapped, z), 1)
        scale_weights = self.mlp_scale(scale_features)
        scale_weights = F.normalize(scale_weights, p=2)

        w0 = scale_weights[:, 0].reshape(n_x, 1).repeat(1, 10)
        w1 = scale_weights[:, 1].reshape(n_x, 1).repeat(1, 10)
        w2 = scale_weights[:, 2].reshape(n_x, 1).repeat(1, 10)
        w3 = scale_weights[:, 3].reshape(n_x, 1).repeat(1, 10)
        w4 = scale_weights[:, 4].reshape(n_x, 1).repeat(1, 10)
        fused_all = w0 * z1_mapped + w1 * z2_mapped + w2 * z3_mapped + w3 * z4_mapped + w4 * z

        predict = F.softmax(self.agcn_z(fused_all, adj, active=False), dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_dtma(dataset):
    print(f"\n[Training Start] Model loaded to device: {device}")
    model = DTMA(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(f"Model device: {next(model.parameters()).device}")

    attention_modules = [model.mlp1, model.mlp2, model.mlp3, model.mlp_scale]
    attention_params = []
    for module in attention_modules:
        attention_params.extend(list(module.parameters()))
    attention_param_ids = {id(p) for p in attention_params}

    main_params = []
    for p in model.parameters():
        if id(p) not in attention_param_ids:
            main_params.append(p)

    optimizer = Adam([
        {'params': main_params, 'lr': args.lr},
        {'params': attention_params, 'lr': args.lr * 1.2}
    ], lr=args.lr)

    dynamic_k = load_dynamic_k_value(args.name)
    graph_path = f'graph/{args.name}{dynamic_k}_graph.txt'
    print(f"Using graph file: {graph_path}")

    adj = load_graph(args.name, k=dynamic_k, graph_path=graph_path)
    adj = adj.to(device)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    train_start_time = time.time()
    for epoch in range(200):
        if epoch % 1 == 0:
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)
            res2 = pred.data.cpu().numpy().argmax(1)
            res3 = p.data.cpu().numpy().argmax(1)
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _ = model(data, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        loss = 0.1 * kl_loss + 0.05 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_train_time = (time.time() - train_start_time) / 60
    print(f"\n[Training End] Total time: {total_train_time:.2f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DTMA Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='dblp.pkl')

    args = parser.parse_args()
    args.name = 'dblp'
    args.pretrain_path = 'data/dblp.pkl'
    dataset = load_data(args.name)

    args.n_clusters = 4
    args.n_input = 334
    args.n_z = 10

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(f"=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda if args.cuda else 'Disabled'}")
    print(f"GPU Available: {args.cuda}")
    print(f"Device: {device}")
    if args.cuda:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"=" * 50)
    print(f"\nTraining Args: {args}")
    train_dtma(dataset)