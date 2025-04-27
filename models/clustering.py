import numpy as np
from typing import Optional, Literal
import hdbscan
from finch import FINCH
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


def clustering_methods(
        z_np: np.ndarray,
        *,
        method: Literal['kmeans', 'hdbscan', 'finch', 'spectral'] = 'kmeans',
        prev_labels: Optional[np.ndarray] = None,
        random_state: int = 0,
        **kwargs
) -> np.ndarray:
    """
    공통 레이블 생성 래퍼
    ---------------------------------------------------------
    Parameters
    ----------
    z_np : (n, d) np.ndarray
        임베딩 (encoder output 등)
    method : {'kmeans', 'hdbscan', 'finch', 'spectral'}
        클러스터러 선택
    prev_labels : np.ndarray or None
        warm-start 초기 중심을 위한 이전 라벨 (K-means 전용)
    kwargs : 기타 클러스터러 별 하이퍼파라미터

    Returns
    -------
    labels : (n,) ndarray of int   (HDBSCAN 노이즈는 -1)
    """
    n = len(z_np)

    # ---------------- K-means ---------------- #
    if method == 'kmeans':
        K = kwargs.get('n_clusters', int(np.sqrt(n)))
        init = 'k-means++' # warm start로 변경해야함!
        if prev_labels is not None and len(np.unique(prev_labels)) == K:
            # warm-start centroid 초기화
            centroids = np.stack([z_np[prev_labels == k].mean(0)
                                  for k in range(K)])
            init = centroids

        km = KMeans(
            n_clusters=K,
            n_init=1 if isinstance(init, np.ndarray) else 20,
            init=init,
            random_state=random_state).fit(z_np)
        return km.labels_

    # ---------------- HDBSCAN --------------- #
    elif method == 'hdbscan':
        min_cluster_size = kwargs.get('min_cluster_size', int(np.sqrt(n)))
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=kwargs.get('min_samples', None),
            metric=kwargs.get('metric', 'euclidean'),
            core_dist_n_jobs=kwargs.get('n_jobs', 1)
        ).fit(z_np)
        return hdb.labels_        # 노이즈는 -1

    # ---------------- FINCH ----------------- #
    elif method == 'finch':
        # FINCH는 거리행렬 필요
        D = pairwise_distances(z_np)
        c, num_clust, _ = FINCH(D, initial_rank=kwargs.get('k', 20))
        # 0번째 partition 사용 (가장 purity 높음)
        return c[:, 0]

    # -------------- Spectral ---------------- #
    elif method == 'spectral':
        K = kwargs.get('n_clusters', int(np.sqrt(n)))
        sc = SpectralClustering(
            n_clusters=K,
            affinity=kwargs.get('affinity', 'nearest_neighbors'),
            random_state=random_state
        ).fit(z_np)
        return sc.labels_

    else:
        raise ValueError(f"Unknown method: {method}")
    

""" 
    -> warm start with KMeans
    # first round
    km = KMeans(K, n_init=1, init='k-means++', random_state=0)
    labels = km.fit_predict(X_t0)
    prev_centroids = km.cluster_centers_

    # subsequent round r
    km = KMeans(
            n_clusters       = K,
            init             = prev_centroids,   # <- warm start
            n_init           = 1,                # *must* be 1 when init is array
            max_iter         = 100,
            random_state     = 0)

    labels = km.fit_predict(X_tr)               # X_tr = embeddings of round r
    prev_centroids = km.cluster_centers_
"""