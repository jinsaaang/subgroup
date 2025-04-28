"""
triplet_autoencoder.py
────────────────────────────────────────────────────────
Two-stage AE → Triplet fine-tune with easy→hard schedule
and adaptive re-clustering (RF-AE backbone)
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random, time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils.miner_stats import count_pair_masks
from models.clustering import clustering_methods

# ------------------ Utility ------------------ #
def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

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
# ------------------ Triplet Miner (Miner Version 1) ------------------ #
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
                print(f"Anchor {i} has no positive sample!")
            neg_cand = torch.where(neg_mask[i])[0]
            if len(neg_cand) == 0:
                neg_cand = torch.tensor([i])
                print(f"Anchor {i} has no negative sample!")
            pos_idx.append(int(pos_cand[torch.randint(0, len(pos_cand), (1,))]))
            neg_idx.append(int(neg_cand[torch.randint(0, len(neg_cand), (1,))]))
        return idx.to(device), torch.tensor(pos_idx, device=device), \
               torch.tensor(neg_idx, device=device)
    
# ------------------ Triplet Miner with RF (Miner Version 2) ------------------ #
class LeafClusterMiner:
    """
    Triplet miner using (1) RF leaf co-occurrence, (2) cluster assignment, (3) ground-truth label.

    Parameters
    ----------
    y_true      : (n,) array-like of int/str     - class labels
    leaf_mat    : (n, n_trees) int ndarray       - leaf IDs per tree
    clusters    : (n,) array-like of int         - cluster IDs (e.g., KMeans)
    """

    def __init__(self,
                 y_true: np.ndarray,
                 leaf_mat: np.ndarray,
                 clusters: np.ndarray):
        self.update_masks(y_true, leaf_mat, clusters)

    # ------------------------------------------------------------------ #
    def update_masks(self, y_true, leaf_mat, clusters):
        """Pre-compute boolean masks on CPU."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(clusters, torch.Tensor):
            clusters = clusters.cpu().numpy()

        # 1) same leaf (at least one tree)
        same_leaf = (leaf_mat[:, None, :] == leaf_mat[None, :, :]).any(axis=2)
        # 2) same cluster
        same_cluster = (clusters[:, None] == clusters[None, :])
        # 3) same label
        same_y = (y_true[:, None] == y_true[None, :])

        # --- masks ----------------------------------------------------- #
        self.EP = torch.from_numpy(same_leaf  & same_cluster & same_y)      # easy +
        self.EN = torch.from_numpy(~same_leaf & ~same_cluster & ~same_y)    # easy −
        self.HP = torch.from_numpy(same_leaf  & same_cluster & ~same_y)     # hard − (same leaf diff label)
        self.HN = torch.from_numpy(same_leaf & ~same_cluster & same_y)    # hard -

    # ------------------------------------------------------------------ #
    def sample_batch(self,
                     idx: torch.Tensor,
                     mode: str = "easy",
                     device: str = "cpu"):
        """
        mode
        ----
        "easy" : EP  vs EN
        "hard" : HP1|HP2 vs EN
        "full" : (EP|HP1) vs (HP2|EN) 
        """
        if mode == "easy":
            pos_mask = self.EP
            neg_mask = self.EN
        elif mode == "hard":
            pos_mask = self.HP  
            neg_mask = self.HN     
        elif mode == "full":
            pos_mask = self.EP | self.HP
            neg_mask = self.EN | self.HN
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # restrict to current mini-batch indices
        pos_mask = pos_mask[idx][:, idx]  # (B,B)
        neg_mask = neg_mask[idx][:, idx]

        B = idx.size(0)
        pos_idx, neg_idx = [], []
        for i in range(B):
            pos_cand = torch.nonzero(pos_mask[i], as_tuple=False).squeeze(1)
            if pos_cand.numel() == 0:
                pos_cand = torch.tensor([i])          # fallback self
                # print(f"Anchor {i} has no positive sample!")
            neg_cand = torch.nonzero(neg_mask[i], as_tuple=False).squeeze(1)
            if neg_cand.numel() == 0:
                neg_cand = torch.tensor([i])
                # print(f"Anchor {i} has no negative sample!")
            pos_idx.append(int(pos_cand[torch.randint(0, len(pos_cand), (1,))]))
            neg_idx.append(int(neg_cand[torch.randint(0, len(neg_cand), (1,))]))

        return (idx.to(device),
                torch.tensor(pos_idx, device=device, dtype=torch.long),
                torch.tensor(neg_idx, device=device, dtype=torch.long))
    
# ------------------ Loss ------------------ #
def triplet_loss(za, zp, zn, margin=0.5): # margin...
    d_pos = F.pairwise_distance(za, zp, p=2) # metric L2 or cosine 
    d_neg = F.pairwise_distance(za, zn, p=2)
    # d_pos = 1.0 - F.cosine_similarity(za, zp, dim=1, eps=1e-8)
    # d_neg = 1.0 - F.cosine_similarity(za, zn, dim=1, eps=1e-8)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss

# ------------------ Margin Scheduler ------------------ #
class MarginScheduler:
    """
    Simple linear scheduler for the triplet-margin hyper-parameter.
    Use `.step_epoch()` at each epoch *or* `.step_round()` at each
    reclustering round, then read `.m` to obtain the current margin.
    """
    def __init__(self,
                 m_start: float = 0.05,
                 m_final: float = 0.5,
                 n_epochs: int | None = None,
                 n_rounds: int | None = None):
        assert (n_epochs is None) ^ (n_rounds is None), \
            "Specify either n_epochs *or* n_rounds (not both)."
        self.m_start, self.m_final = m_start, m_final
        self.n_total = n_epochs if n_epochs is not None else n_rounds
        self.curr = 0
        self.m = m_start
        self.delta = (m_final - m_start) / max(1, self.n_total - 1)

    # call once per training epoch ----------------------------------
    def step_epoch(self):
        if self.n_total is None:
            raise RuntimeError("Scheduler initialised for 'round' mode.")
        self._update()

    # call once per reclustering round ------------------------------
    def step_round(self):
        if self.n_total is None:
            raise RuntimeError("Scheduler initialised for 'epoch' mode.")
        self._update()

    def _update(self):
        if self.curr < self.n_total - 1:
            self.curr += 1
            self.m = self.m_start + self.delta * self.curr

# ------------------ Trainer ------------------ #
class Trainer:
    def __init__(
        self, 
        x_gap: np.ndarray, 
        y_true: np.ndarray,
        z_star: np.ndarray,   # RF-PHATE embeddings (n, z_dim)
        n_clusters: int,
        leaf_mat: np.ndarray,  # (n, n_trees) RF leaf IDs
        margin: 0.3, 
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
        self.margin = margin
        # initial clusters from z_star
        self.n_clusters = n_clusters
        self.initial_clusters, self.initial_centroid = self._kmeans_cluster(z_np=z_star, K=10) 
        self.clusters = self.initial_clusters.copy()
        self.centroid = self.initial_centroid.copy()
        self.leaf_mat = leaf_mat
        # self.miner = EasyHardMiner(self.labels, self.clusters)
        self.miner = LeafClusterMiner(self.labels, self.leaf_mat, self.clusters)
        self.best_clusters = None
        self.best_embeddings = None
        # ------- results -------
        self.nmi_list = None
        self.ari_list = None
    # --------------------------------------------------
    def _kmeans_cluster(self, z_np: np.ndarray, K: int | None = None,
                        prev_centroids: np.ndarray | None = None, random_state: int = 42):
        # Perform K-Means (optionally warm-started) on latent vectors.
        if K is None:
            K = int(np.sqrt(len(z_np)))

        if prev_centroids is not None:
            # Warm-start: single initialisation using previous centres
            km = KMeans(n_clusters=K,
                        init=prev_centroids,
                        n_init=1,
                        max_iter=300,
                        random_state=random_state).fit(z_np)
        else:
            # Faster for large n – still converges to normal k-means solution
            km = MiniBatchKMeans(n_clusters=K,
                                batch_size=1024,
                                n_init=10,
                                random_state=random_state).fit(z_np)

        return km.labels_, km.cluster_centers_
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
        scehduler = MarginScheduler(m_start=0.05, m_final=margin, n_rounds=rounds)
        for ep in range(E_pre):
            losses = self._epoch_step(loader, lambda_r=lambda_r, lambda_g=(1-lambda_r),
                             lambda_t=0, miner=None, mode="easy", margin=self.margin)
            if (ep+1) % 10 == 0:
                print(f"[Pre-train] Epoch {ep+1} | "
                      f"loss={losses['total']:.4f} | "
                      f"recon={losses['recon']:.4f} | "
                      f"geom={losses['geom']:.4f}")
        print("[Pre-train]✨ Pre-train done")
        
        # ---- Stage 2: Loop ----
        prev_clusters = self.initial_clusters.copy()
        prev_centroids = self.initial_centroid.copy()
        best_nmi = 0.0
        nmi_list = []
        ari_list = []
        best_clusters = self.clusters.copy()
        best_embeddings = self.model.encoder(self.X).detach().cpu().numpy()

        for r in range(1, rounds+1):
            # fine-tune with easy→hard curriculum
            for t in range(T):
                mode = "easy" if t < T//2 else "hard"
                # losses = self._epoch_step(loader, lambda_r*0.5, (1-lambda_r)*0.5, 0.5, self.miner, mode, margin)
                losses = self._epoch_step(loader, 0, 0, 1, self.miner, mode, margin) # only triplet loss
                if (t+1) % 10 == 0:
                    print(f"[Round {r}][Epoch {t+1}][{mode.upper()} Sample] "
                          f"loss={losses['total']:.4f} | "
                          f"recon={losses['recon']:.4f} | "
                          f"geom={losses['geom']:.4f} | "
                          f"triplet={losses['trip']:.4f}")
            # recluster
            z_all = self.model.encoder(self.X).detach().cpu().numpy()
            self.clusters, self.centroid = self._kmeans_cluster(z_np=z_all, K=10, prev_centroids=prev_centroids)
            self.miner.update_masks(self.labels, self.leaf_mat, self.clusters)
            nmi = normalized_mutual_info_score(prev_clusters, self.clusters)
            ari = adjusted_rand_score(prev_clusters, self.clusters)
            nmi_list.append(nmi)
            ari_list.append(ari)
            print(f"[Round {r}] NMI(prev,new)={nmi:.4f} | ARI(prev,new)={ari:.4f}") 
            if nmi > best_nmi:
                best_nmi = nmi
                best_clusters = self.clusters.copy()
                best_embeddings = z_all.copy()  # 현재 임베딩 저장

            if nmi > 1 - tol_nmi:
                print("🧨 Converged: cluster change small")
                break
            prev_clusters = self.clusters.copy()
            prev_centroids = self.centroid.copy()

            n_ep, n_en, n_hp, n_hn = count_pair_masks(self.miner)
            print(f"[Round {r}] EP:{n_ep:,} EN:{n_en:,} "
                f"HP:{n_hp:,} HN:{n_hn:,}")
        
        self.best_clusters = best_clusters
        self.best_embeddings = best_embeddings
        self.nmi_list = nmi_list
        self.ari_list = ari_list

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
