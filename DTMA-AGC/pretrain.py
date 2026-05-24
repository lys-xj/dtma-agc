import warnings
import os
warnings.filterwarnings("ignore", category=RuntimeWarning, message="scipy._lib.messagestream")
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores")


import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from evaluation import eva
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device：{device}")


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

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        scaler = StandardScaler()
        self.x = scaler.fit_transform(data)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def pretrain_ae(model, dataset, y, n_clusters, save_weight_path):
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)


    for epoch in range(50):
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            x_all = torch.from_numpy(dataset.x).float().to(device)
            x_bar, z = model(x_all)
            recon_loss = F.mse_loss(x_bar, x_all)
            print(f"Epoch {epoch:2d} | recon loss：{recon_loss:.6f}")

            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, f"AE-Pretrain-Epoch-{epoch}")

        torch.save(model.state_dict(), save_weight_path)
        print(f"weights saved to：{save_weight_path}\n")

if __name__ == "__main__":
    # target_dataset = "acm"
    # target_dataset = "reut"
    target_dataset = "dblp"
    data_x_path = f"data/{target_dataset}.txt"
    data_y_path = f"data/{target_dataset}_label.txt"
    save_weight_path = f"data/{target_dataset}.pkl"

    print("="*50)
    print(f"start pretrain AE | dataset：{target_dataset} | device：{device}")
    print("="*50)

    x = np.loadtxt(data_x_path, dtype=float)
    y = np.loadtxt(data_y_path, dtype=int)
    dataset = LoadDataset(x)
    n_input = x.shape[1]
    n_clusters = len(np.unique(y))

    print(f"dataset loaded！samples：{len(dataset)} | feature dim：{n_input} | clusters：{n_clusters}")

    model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=n_input,
        n_z=10
    ).to(device)

    pretrain_ae(model, dataset, y, n_clusters, save_weight_path)

    print("="*50)
    print("AE pretrain finished! final weights saved to：", save_weight_path)
    print("="*50)