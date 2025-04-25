import yaml
import numpy as np
import pandas as pd
import torch
import sklearn
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# from models.vime_encoder import VIMEEncoder
# from models.classifier import XGBoostClassifier, NODE, TabNet
# from models.soft_cluster import DPGMM
from utils.loader import load_data
# from utils.visualizer import plot_sillhouette, plot_dbindex, plot_umap, plot_group_acc
from rfae.autoencoder import Trainer

import rfphate
import seaborn as sns

# Load Train Arguments
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using {device}")

timestamp = datetime.now().strftime("%m%d%H%M")
exp_dir = f'./results/exp_{timestamp}'
os.makedirs(exp_dir, exist_ok=True)
dataset = config['dataset'].lower()

batch_size = config['batch_size']
lr = config['learning_rate']
random_state = config['random_state']

e_pre = config['n_pre_epochs']
rounds = config['n_rounds']
epochs = config['n_epochs']

# Load Data
# train_loader, test_loader, train_df = load_data(dataset) # valid 를 없애야함, numpy로 받아오기기
# X_train = train_loader.dataset.features
# y_train = train_loader.dataset.labels
# n_train = X_train.shape[0]
# input_dim = X_train.shape[1]

data = rfphate.load_data('titanic')
X_train, y_train = rfphate.dataprep(data) # numpy array
print("[INFO] Data loaded")

# RF-PHATE
rfphate_op = rfphate.RFPHATE(random_state = 42, n_landmark=100)
z_geom = rfphate_op.fit_transform(X_train, y_train)
p_land, _ = rfphate_op.dimension_reduction() # p_land: (n, land), cluster_label
print("[INFO] RF-PHATE done")

# Train Autoencoder -> p_land: (land, land), y_train: (n,), z_geom: (2, 2)
trainer = Trainer(p_land, y_train, z_geom, n_clusters=10, lr=lr, batch=batch_size, device=device)
clusters, z_hat = trainer.train(E_pre=e_pre, rounds=rounds, T=epochs)
# train_df['cluster'] = clusters
data['cluster'] = clusters
initial_clusters = trainer.initial_clusters
print("[INFO] Autoencoder training done")

# print(train_df.head())
print(data.head())
print(stop)

# Visualize - sillhoute score (Original Embedding vs Triplet Embedding)
# plot_sillhouette(z_geom, z_hat)
# plot_dbindex(z_geom, z_hat)
# plot_umap(z_geom, z_hat)
# plot_group_acc(train_loader, test_loader, initial_clusters, clusters) # initial clustering?

# plot_error_dist(initial_clusters, clusters)


