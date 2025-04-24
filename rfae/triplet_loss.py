from typing import Tuple
import torch, torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#               ───  Mask Utilities (NxN boolean) ───
# ------------------------------------------------------------------ #
def build_masks(clusters: np.ndarray,
                labels:   np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Returns 4 boolean masks (NxN):

    EP : same_c & same_y
    HP : diff_c & same_y
    HN : same_c & diff_y
    EN : diff_c & diff_y
    """
    same_c = clusters[:, None] == clusters[None, :]
    same_y = labels[:,   None] == labels[None,   :]

    EP =  same_c &  same_y
    HP = ~same_c &  same_y
    HN =  same_c & ~same_y
    EN = ~same_c & ~same_y
    np.fill_diagonal(EP, False)                        # self pair 제거
    return EP, HP, HN, EN


# ------------------------------------------------------------------ #
#                         Batch-Hard Mining
# ------------------------------------------------------------------ #
def batch_hard(dist: torch.Tensor,
               pos_mask: torch.Tensor,
               neg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dist : (B,B) distance matrix (larger = 더 멂)
    pos_mask / neg_mask : boolean  (True 위치만 유효)
    """
    # hardest positive = max distance among positives
    p = dist.clone()
    p[~pos_mask] = -1e9
    d_pos, _ = p.max(dim=1)

    # hardest negative = min distance among negatives
    n = dist.clone()
    n[~neg_mask] =  1e9
    d_neg, _ = n.min(dim=1)
    return d_pos, d_neg


# ------------------------------------------------------------------ #
#                       Scheduled Triplet Loss
# ------------------------------------------------------------------ #
class ScheduledTripletLoss(torch.nn.Module):
    """
    Parameters
    ----------
    margin_init / margin_final : float
        Annealing range   α(epoch=0) → α(epoch=total_epochs)
    T1, T2 : int
        Curriculum switch (EP+EN) → (HP 추가) → (HN 추가)
    l_hp, l_hn : float
        λ 가중치 (0~1) · 스케줄 동안 선형 증가
    """

    def __init__(self,
                 margin_init:  float = 0.5,
                 margin_final: float = 0.2,
                 total_epochs: int   = 100,
                 T1: int = 10,
                 T2: int = 30,
                 l_hp: float = 1.0,
                 l_hn: float = 1.0):
        super().__init__()
        self.m0, self.m1 = margin_init, margin_final
        self.E = total_epochs
        self.T1, self.T2 = T1, T2
        self.l_hp, self.l_hn = l_hp, l_hn

    # ---------- public forward ---------- #
    def forward(self,
                embeddings: torch.Tensor,        # (B,d)
                clusters:   torch.Tensor,        # (B,) int64
                labels:     torch.Tensor,        # (B,) int64
                epoch:      int,
                dist_mat: torch.Tensor = None    # optional pre-computed
                ) -> torch.Tensor:

        if dist_mat is None:
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # build masks  (numpy→torch)
        EP, HP, HN, EN = build_masks(clusters.cpu().numpy(),
                                     labels.cpu().numpy())
        device = embeddings.device
        EP = torch.from_numpy(EP).to(device)
        HP = torch.from_numpy(HP).to(device)
        HN = torch.from_numpy(HN).to(device)
        EN = torch.from_numpy(EN).to(device)

        # curriculum 스케줄
        pos_mask = EP.clone()
        neg_mask = EN.clone()

        # add HP after T1
        if epoch >= self.T1:
            pos_mask |= HP

        # add HN after T2
        if epoch >= self.T2:
            neg_mask |= HN

        # batch-hard mining
        d_pos, d_neg = batch_hard(dist_mat, pos_mask, neg_mask)

        # margin annealing
        margin = self.m0 - (self.m0 - self.m1) * min(epoch, self.E) / self.E

        # λ weight schedule (linear ramp)
        hp_w = self._linear_ramp(epoch, self.T1, self.l_hp)
        hn_w = self._linear_ramp(epoch, self.T2, self.l_hn)

        loss = F.relu(d_pos - d_neg + margin)          # 기본
        loss = loss.mean()

        # add extra penalties (optional)
        if epoch >= self.T1:
            # HP 전용 add-on (양성 수축 강제)
            loss += hp_w * d_pos.mean()
        if epoch >= self.T2:
            # HN 전용 add-on (음성 분리 강제)
            loss += hn_w * F.relu(-d_neg + margin).mean()

        return loss

    # ---------- util ---------- #
    @staticmethod
    def _linear_ramp(epoch, start, target):
        """epoch ≥ start 일 때 0→target 선형 상승"""
        if epoch < start:
            return 0.0
        return target * (epoch - start) / max(1, (start))
    

"""
# Example usage:
from triplet_loss import BatchHardTripletLoss

triplet_loss = BatchHardTripletLoss(margin=0.3)

z = encoder(x)  # (B,d)
loss_tri = triplet_loss(z, cluster_id, y_true)
total_loss = λ1*recon + λ2*geo + λ3*loss_tri
"""