import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans
from typing import Dict


def calculate_clustering_metrics(
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: int = None
) -> Dict[str, float]:
    """
    计算聚类评估指标

    Args:
        embeddings: 嵌入向量，形状 [n_samples, n_features]
        true_labels: 真实标签，形状 [n_samples]
        n_clusters: 聚类数量，如果为None则使用真实标签的类别数

    Returns:
        metrics: 包含所有指标的字典
    """
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    # 计算指标
    metrics = {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "Silhouette": silhouette_score(embeddings, pred_labels),
        "Calinski-Harabasz": calinski_harabasz_score(embeddings, pred_labels),
        "Davies-Bouldin": davies_bouldin_score(embeddings, pred_labels)
    }

    return metrics