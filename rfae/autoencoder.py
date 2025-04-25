"""
trainer_triplet_loop.py
────────────────────────────────────────────────────────
Two-stage AE → Triplet fine-tune with easy→hard schedule
and adaptive re-clustering (RF-AE backbone)
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random, time
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from models.soft_cluster import DPGMM

# ------------------ Utility ------------------ #
def set_seed(sd=0):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
    return 1 - torch.mm(a, b.t())

# ------------------ Model ------------------ #
class RFAE(nn.Module):
    """Simple RF-AE encoder/decoder"""
    def __init__(self, in_dim: int, z_dim: int = 32, hid: int = 256): # 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid), 
            nn.ReLU(),
            nn.Linear(hid, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hid), 
            nn.ReLU(),
            nn.Linear(hid, in_dim), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon  # recon already prob. simplex
# ------------------ Triplet Miner ------------------ #
class EasyHardMiner:
    """
    Maintains masks & sampling strategy (easy→hard schedule)
    """
    def __init__(self, y_true: np.ndarray, clusters: np.ndarray):
        self.update_masks(y_true, clusters)
    # -----------------------------------------------------------------
    def update_masks(self, y_true, clusters):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(clusters, torch.Tensor):
            clusters = clusters.cpu().numpy()

        y = y_true[:, None]
        c = clusters[:, None]
        self.same_y  = torch.from_numpy((y == y.T)).bool()
        self.same_c  = torch.from_numpy((c == c.T)).bool()
        # Masks
        self.EP = self.same_y  & self.same_c          # easy+
        self.EN = (~self.same_y) & (~self.same_c)     # easy-
        self.HP = self.same_y  & (~self.same_c)       # hard+
        self.HN = (~self.same_y) & self.same_c        # hard-
    # -----------------------------------------------------------------
    def sample_batch(
        self, idx: torch.Tensor, mode: str = "easy", device="cpu"
    ):
        """
        Return anchor, pos, neg index lists (torch.LongTensor)
        mode = easy -> EP+EN / hard -> HP+HN / full -> mix
        """
        masks = {"easy": (self.EP, self.EN),
                 "hard": (self.HP, self.HN),
                 "full": (self.EP | self.HP, self.EN | self.HN)}
        pos_mask, neg_mask = masks[mode]
        pos_mask = pos_mask[idx][:, idx]       # (B,B)
        neg_mask = neg_mask[idx][:, idx]

        # For each anchor row, randomly pick 1 pos / 1 neg
        B = idx.shape[0]
        pos_idx = []
        neg_idx = []
        for i in range(B):
            pos_cand = torch.where(pos_mask[i])[0]
            if len(pos_cand) == 0:
                pos_cand = torch.tensor([i])   # fallback self
            neg_cand = torch.where(neg_mask[i])[0]
            if len(neg_cand) == 0:
                neg_cand = torch.tensor([i])
            pos_idx.append(int(pos_cand[torch.randint(0, len(pos_cand), (1,))]))
            neg_idx.append(int(neg_cand[torch.randint(0, len(neg_cand), (1,))]))
        return idx.to(device), torch.tensor(pos_idx, device=device), \
               torch.tensor(neg_idx, device=device)

# ------------------ Loss ------------------ #
def triplet_loss(za, zp, zn, margin=0.2): # margin...
    d_pos = F.pairwise_distance(za, zp, p=2) # metric L2 or cosine 
    d_neg = F.pairwise_distance(za, zn, p=2)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss

# ------------------ Trainer ------------------ #
class Trainer:
    def __init__(
        self, x_gap: np.ndarray, 
        y_true: np.ndarray,
        z_star: np.ndarray,   # RF-PHATE embeddings (n, z_dim)
        n_clusters: int,
        lr=1e-3, batch=256, device="cuda"
    ):
        set_seed()
        self.device = torch.device(device)
        self.X = torch.tensor(x_gap, dtype=torch.float32).to(device)
        self.z_star = torch.tensor(z_star, dtype=torch.float32).to(device)
        self.labels = y_true
        # ------- model -------
        self.model = RFAE(in_dim=x_gap.shape[1], z_dim=z_star.shape[1]).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr)
        self.batch = batch
        # initial clusters from z_star
        self.n_clusters = n_clusters
        self.initial_clusters = self._kmeans_cluster(z_star, n_clusters) 
        self.clusters = self._kmeans_cluster(z_star, n_clusters)
        self.miner = EasyHardMiner(self.labels, self.clusters)
        self.best_clusters = None
        self.best_embeddings = None
    # --------------------------------------------------
    def _kmeans_cluster(self, z_np, K=None):
        # clustering은 추후 교체 예정
        if K is None:
            K = int(np.sqrt(len(z_np)))
        km = KMeans(K, n_init=20, random_state=0).fit(z_np)
        return km.labels_
    # --------------------------------------------------
    def reconstruction_loss(self, p, p_hat):
        p = p.clamp(min=1e-8)         
        p_hat = p_hat.clamp(min=1e-8)   
        kl = (p * (p.log() - p_hat.log())).sum(dim=1).mean()
        return kl
    # --------------------------------------------------
    def train(
        self, E_pre=50, rounds=5, T=10,
        lambda_r=1e-3, lambda_t=1.0,
        margin=0.2, tol_nmi=1e-3
    ):
        loader = DataLoader(
            TensorDataset(self.X, self.z_star),
            batch_size=self.batch, shuffle=True, drop_last=True
        )
        # ---- Stage 1: Pre-train AE ----
        for ep in range(E_pre):
            losses = self._epoch_step(loader, lambda_r=lambda_r, lambda_g=(1-lambda_r),
                             lambda_t=0, miner=None, mode="easy", margin=margin)
            if (ep+1) % 10 == 0:
                print(f"[Pre-train] Epoch {ep+1} | "
                      f"loss={losses['total']:.4f} | "
                      f"recon={losses['recon']:.4f} | "
                      f"geom={losses['geom']:.4f}")
        print("[Pre-train]✨ Pre-train done")
        
        # ---- Stage 2: Loop ----
        prev_labels = self.clusters.copy()
        best_nmi = 0.0
        best_clusters = self.clusters.copy()
        best_embeddings = self.model.encoder(self.X).detach().cpu().numpy()

        for r in range(1, rounds+1):
            # fine-tune with easy→hard curriculum
            for t in range(T):
                mode = "easy" if t < T//5 else "hard"
                losses = self._epoch_step(loader, lambda_r*0.1, (1-lambda_r)*0.1, # error
                                 0.9, self.miner, mode, margin)
                if (t+1) % 5 == 0:
                    print(f"[Round {r}][Epoch {t+1}] "
                          f"loss={losses['total']:.4f} | "
                          f"recon={losses['recon']:.4f} | "
                          f"geom={losses['geom']:.4f} | "
                          f"triplet={losses['trip']:.4f}")
            # recluster
            z_all = self.model.encoder(self.X).detach().cpu().numpy()
            self.clusters = self._kmeans_cluster(z_all, self.n_clusters)
            self.miner.update_masks(self.labels, self.clusters)
            nmi = normalized_mutual_info_score(prev_labels, self.clusters)
            print(f"[Round {r}] NMI(prev,new)={nmi:.4f}")
            if nmi > best_nmi:
                best_nmi = nmi
                best_clusters = self.clusters.copy()
                best_embeddings = z_all.copy()  # 현재 임베딩 저장

            if nmi > 1 - tol_nmi:
                print("🧨 Converged: cluster change small")
                break
            prev_labels = self.clusters.copy()
        
        self.best_clusters = best_clusters
        self.best_embeddings = best_embeddings

        return best_clusters, best_embeddings
    # --------------------------------------------------
    def _epoch_step(self, loader, lambda_r, lambda_g, lambda_t,
                    miner: EasyHardMiner, mode, margin):
        self.model.train()
        total_loss = 0.0
        recon_loss = 0.0
        geom_loss = 0.0
        trip_loss = 0.0
        n_batches = 0

        for xb, zb in loader: # xb: (n, land), zb: (n, z_dim)
            xb, zb = xb.to(self.device), zb.to(self.device)
            idx = torch.arange(xb.size(0), device=self.device)
            z, recon = self.model(xb)
            loss_recon = lambda_r * self.reconstruction_loss(xb, recon)
            loss_geom = lambda_g * F.mse_loss(z, zb)
            loss =  loss_recon + loss_geom

            loss_triplet = 0.0
            if lambda_t > 0 and miner is not None:
                anc, pos_i, neg_i = miner.sample_batch(idx.cpu(), mode, device=self.device)
                loss_triplet = lambda_t * triplet_loss(z[anc], z[pos_i], z[neg_i], margin)
                loss += loss_triplet
            
            self.opt.zero_grad(); loss.backward(); self.opt.step()

            # Accumulate losses
            total_loss += loss.item()
            recon_loss += loss_recon.item()
            geom_loss += loss_geom.item()
            trip_loss += loss_triplet.item() if lambda_t > 0 else 0.0
            n_batches += 1
        
        # Return average losses (losses)
        return {
        'total': total_loss / n_batches,
        'recon': recon_loss / n_batches,
        'geom': geom_loss / n_batches,
        'trip': trip_loss / n_batches
        }

# X_prob : RF-GAP 확률벡터 (n,m)  , y : (n,)  , z_star : RF-PHATE (n,2)
# trainer = Trainer(X_prob, y, z_star, lr=1e-3, batch=512, device="cuda")
# trainer.train(E_pre=40, rounds=6, T=20)
# embedding = trainer.model.encoder(torch.tensor(X_prob).to("cuda")).detach().cpu().numpy()
