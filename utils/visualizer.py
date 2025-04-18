import umap.umap_ as umap
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def visualize_umap(z_valid, group_labels, error, iter=0, dir="."):
    """
    UMAP 시각화 후 저장 (group color + error 컬러 intensity)

    Args:
        z_valid (tensor or np.ndarray): (N, D) embedding
        group_labels (tensor or np.ndarray): (N,) hard cluster labels
        error (tensor or np.ndarray): (N,) or (N, 1) validation error
        iter (int): current iteration (used in filename)
        dir (str): directory to save the image
    """
    # tensor → numpy 변환
    if isinstance(z_valid, torch.Tensor):
        z_valid = z_valid.detach().cpu().numpy()
    if isinstance(group_labels, torch.Tensor):
        group_labels = group_labels.detach().cpu().numpy()
    if isinstance(error, torch.Tensor):
        error = error.detach().cpu().numpy()

    error = error.squeeze()

    # UMAP으로 2D 임베딩
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = reducer.fit_transform(z_valid)  # (N, 2)

    # 플롯
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(umap_result[:, 0], umap_result[:, 1],
                     c=error, cmap='viridis', s=20, alpha=0.9,
                     edgecolors='k', linewidths=0.3)
    plt.title(f"UMAP - Iter {iter} (Colored by Error)")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.colorbar(sc, label='Adjusted Error')
    
    # 그룹 경계 시각화를 위해 라벨도 같이 출력
    for g in np.unique(group_labels):
        mask = (group_labels == g)
        x, y = umap_result[mask, 0].mean(), umap_result[mask, 1].mean()
        plt.text(x, y, str(g), fontsize=8, color='white',
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.4, lw=0))

    # 저장
    filename = os.path.join(dir, f"umap_{iter:02d}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
