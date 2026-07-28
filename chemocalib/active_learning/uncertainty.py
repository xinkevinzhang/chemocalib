"""
不确定性采样器 —— 主动学习核心
=====================================
基于 MB-PLS 残差空间的 uncertainty sampling,
选择"下一个该做哪个双敲除"。

策略:
  1. Residual-based uncertainty (残差范数)
  2. Ensemble disagreement (多模型分歧)
  3. Density-weighted sampling (密度加权)
  4. Hybrid: 结合 VIP 重要性与不确定性

这是 DoE 老本行的高光时刻 ——
用化学计量学的残差空间指导实验设计。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.spatial.distance import cdist


class UncertaintySampler:
    """
    基于不确定性的主动学习采样器

    参数
    ----
    strategy : str
        采样策略:
        - "residual": 残差 L2 范数 (默认)
        - "entropy": 预测熵估计
        - "diversity": 多样性最大化
        - "hybrid": 混合策略 (残差 + 多样性)
    """

    def __init__(self, strategy: str = "residual", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # 历史记录
        self.history_samples: List[int] = []
        self.history_scores: List[float] = []
        self._uncertainty_cache: Optional[np.ndarray] = None

    def compute_uncertainty(
        self,
        residuals: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        计算样本不确定性

        参数
        ----
        residuals : list of np.ndarray
            各块的残差矩阵 [R1, R2, ..., Rk], 每个 Ri shape (n_samples, n_features)
        weights : list of float, optional
            各块的权重

        返回
        ----
        uncertainty : np.ndarray (n_samples,)
        """
        n_samples = residuals[0].shape[0]
        n_blocks = len(residuals)

        if weights is None:
            weights = [1.0 / n_blocks] * n_blocks

        uncertainty = np.zeros(n_samples)

        if self.strategy == "residual":
            # L2 范数加权
            for i, R in enumerate(residuals):
                uncertainty += weights[i] * np.linalg.norm(R, axis=1)

        elif self.strategy == "entropy":
            # 用残差方差作 entropy proxy
            for i, R in enumerate(residuals):
                var = np.var(R, axis=1)
                # 归一化到 [0, 1]
                vmax = var.max()
                if vmax > 0:
                    var = var / vmax
                uncertainty += weights[i] * var

        elif self.strategy == "diversity":
            # 与已选样本的距离最大化 (需要已选样本)
            if not self.history_samples:
                uncertainty = np.ones(n_samples)
            else:
                # 计算每个候选到最近已选样本的距离
                all_residuals = np.hstack([R for R in residuals])
                selected_vecs = all_residuals[self.history_samples]
                dists = cdist(all_residuals, selected_vecs, metric="euclidean")
                uncertainty = np.min(dists, axis=1)

        elif self.strategy == "hybrid":
            # 残差 + 多样性
            u_residual = np.zeros(n_samples)
            for i, R in enumerate(residuals):
                u_residual += weights[i] * np.linalg.norm(R, axis=1)

            if not self.history_samples:
                u_diversity = np.ones(n_samples)
            else:
                all_residuals = np.hstack([R for R in residuals])
                selected_vecs = all_residuals[self.history_samples]
                dists = cdist(all_residuals, selected_vecs, metric="euclidean")
                u_diversity = np.min(dists, axis=1) / (np.max(dists) + 1e-8)

            uncertainty = (
                0.5 * (u_residual / (u_residual.max() + 1e-8)) + 0.5 * u_diversity
            )

        self._uncertainty_cache = uncertainty
        return uncertainty

    def select_samples(
        self,
        residuals: List[np.ndarray],
        n_select: int = 10,
        candidate_mask: Optional[np.ndarray] = None,
        weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择下一个该测试的样本

        参数
        ----
        residuals : list of np.ndarray
            各块残差
        n_select : int
            选取数量
        candidate_mask : np.ndarray (bool), optional
            候选集掩码
        weights : list of float, optional
            块权重

        返回
        ----
        selected_indices : np.ndarray
            选中的样本索引
        scores : np.ndarray
            对应的不确定性分数
        """
        uncertainty = self.compute_uncertainty(residuals, weights)

        if candidate_mask is not None:
            uncertainty[~candidate_mask] = -np.inf

        selected = np.argsort(uncertainty)[::-1][:n_select]
        scores = uncertainty[selected]

        self.history_samples.extend(selected.tolist())
        self.history_scores.extend(scores.tolist())

        return selected, scores

    def select_double_knockout_candidates(
        self,
        all_gene_pairs: List[Tuple[str, str]],
        pair_features: np.ndarray,
        residuals: List[np.ndarray],
        n_select: int = 10,
        n_pool: int = 200,
    ) -> pd.DataFrame:
        """
        从给定基因对池中选择最优双敲除候选

        这是阶段二的科学卖点:
          "用 MB-PLS 残差空间指导选择下一个双敲除实验"

        参数
        ----
        all_gene_pairs : list of (str, str)
            所有候选基因对
        pair_features : np.ndarray (n_pairs, n_features)
            基因对的特征向量 (如双敲除的表型预测特征)
        residuals : list of np.ndarray
            MB-PLS 残差
        n_select : int
            最终选取数量 (给合作者真做的)
        n_pool : int
            候选池大小 (in-silico 预筛选)

        返回
        ----
        candidates : pd.DataFrame
            columns: [gene_a, gene_b, uncertainty, rank, reason]
        """
        uncertainty = self.compute_uncertainty(residuals)

        # 如果有超过 n_pool 个候选, 先按 uncertainty 预筛选
        if len(all_gene_pairs) > n_pool:
            pool_indices = np.argsort(uncertainty)[::-1][:n_pool]
        else:
            pool_indices = np.arange(len(all_gene_pairs))

        # 用多样性策略从池中再选
        pool_residuals = [
            R[pool_indices] if R.shape[0] > max(pool_indices) else R for R in residuals
        ]

        selected, scores = self.select_samples(
            pool_residuals,
            n_select=n_select,
        )

        # 组装结果
        rows = []
        for rank, (idx, score) in enumerate(zip(selected, scores)):
            pair_idx = pool_indices[idx] if idx < len(pool_indices) else idx
            if pair_idx < len(all_gene_pairs):
                ga, gb = all_gene_pairs[pair_idx]
                rows.append(
                    {
                        "gene_a": ga,
                        "gene_b": gb,
                        "uncertainty": score,
                        "rank": rank + 1,
                        "reason": f"Top-{rank+1} by {self.strategy} uncertainty",
                    }
                )

        return pd.DataFrame(rows)

    def reset_history(self):
        """清除采样历史"""
        self.history_samples = []
        self.history_scores = []
        self._uncertainty_cache = None

    def summary(self) -> str:
        """采样器状态"""
        return (
            f"UncertaintySampler(strategy={self.strategy}, "
            f"samples_selected={len(self.history_samples)})"
        )
