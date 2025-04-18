import yaml
import numpy as np
import pandas as pd
import torch
import gc
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from models.vime_encoder import VIMEEncoder
from models.classifier import XGBoostClassifier, NODE, TabNet
from models.soft_cluster import DPGMM
from utils.loader import load_data
from utils.train import Trainer
from utils.error import compute_error, adjust_error
from utils.visualizer import visualize_umap

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_type = {
    'node': NODE,
    'tabnet': TabNet,
    'xgboost': XGBoostClassifier
}

cluster_type = {
    'hdbscan': 'PlainDPGMM',
    'dpgmm': DPGMM,
    'spectral': 'SoftCluster'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

timestamp = datetime.now().strftime("%m%d%H%M")
exp_dir = f'./results/exp_{timestamp}'
os.makedirs(exp_dir, exist_ok=True)
dataset = config['dataset'].lower()
n_iters = config['n_iters']

train_loader, valid_loader, test_loader, train_df = load_data(dataset)

# Get Latent Representations
X_train = train_loader.dataset.features
X_valid = valid_loader.dataset.features
input_dim = X_train.shape[1]
encoder = VIMEEncoder(input_dim=input_dim, alpha=1.0, hidden_dim=input_dim)
encoder.train_encoder(X_valid)
z_train = encoder.encode(X_train) 
z_valid = encoder.encode(X_valid) 

# Latent Grouping via Soft Clustering
cluster_name = cluster_type[config['cluster']]
cluster_model = cluster_name(config['cluster_params'])
cluster_model.fit(z_valid)
train_group_labels = cluster_model.predict(z_train) # hard cluster labels
valid_group_labels = cluster_model.predict(z_valid) # hard cluster labels
train_group = cluster_model.predict_proba(z_train) # soft cluster probabilities
valid_group = cluster_model.predict_proba(z_valid) # soft cluster probabilities

# ERM Model 
model_name = model_type[config['model']]
model = model_name(input_dim=input_dim, config=config['model_params']).to(device)
trainer = Trainer(model) # loss_fn
trainer.train(train_loader, config['train_params'], device=device)
y_train_true = train_loader.dataset.labels
y_valid_true = valid_loader.dataset.labels

y_train_hat = trainer.predict(train_loader)
y_valid_hat = trainer.predict(valid_loader)

# prediction error
train_error = compute_error(y_train_hat, y_train_true)
valid_error = compute_error(y_valid_hat, y_valid_true)
print("[Train] Model Training Done")

# Visualize cluster labels
visualize_umap(z_valid, valid_group_labels, valid_error, iter=0, dir=exp_dir)
print("[Clustering Done]")

# Iteration
for i in range(n_iters):
    # Error Adjustment via latent group info
    train_error_adj, valid_error_adj = adjust_error(train_error, valid_error, q_train=train_group, q_valid=valid_group, 
                                                    g_train=train_group_labels, y_train=y_train_true, y_valid=y_valid_true,
                                                    alpha=1.0, beta=1.0)

    # Joint Embedding z_[z_j, r^adj_j] (concatenate z_valid and error_adj)
    z_valid_concat = torch.cat((z_valid, valid_error_adj), dim=1)
    z_train_concat = torch.cat((z_train, train_error_adj), dim=1)
    
    # Error-Aware Re-Clustering
    del cluster_model
    gc.collect()

    cluster_model = cluster_name(config['cluster_params']) # reinitialize cluster model
    cluster_model.fit(z_valid_concat)
    
    train_group_labels = cluster_model.predict(z_train_concat) 
    train_group = cluster_model.predict_proba(z_train_concat) # soft cluster probabilities
    valid_group = cluster_model.predict_proba(z_valid_concat) # soft cluster probabilities

    # Visualize cluster labels after error adjustment
    visualize_umap(z_valid, valid_group_labels, valid_error_adj, iter=i+1, dir=exp_dir)
    print(f"[Iteration {i+1} Done]")