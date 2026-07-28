"""
多块 PLS (Multi-Block PLS) —— 阶段二核心模块
=============================================

实现 mixOmics 风格的 MB-PLS 算法，支持:
  - 三块数据对齐: 代谢组 X1, 转录组 X2, 蛋白组 X3
  - 潜变量提取与跨块协方差最大化
  - VIP 评分筛选驱动代谢物
  - 残差空间分析 (供主动学习用)

参考文献:
  - Westerhuis et al. (1998) J. Chemometrics
  - Rohart et al. (2017) PLOS Comp. Biol. (mixOmics)
  - Qin et al. (2001) AIChE Journal (MB-PLS)

用于轻薄本: 纯 numpy + sklearn, 无需 GPU
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MultiBlockPLS:
    """
    多块偏最小二乘 (MB-PLS)

    支持 k 个数据块 (X1, X2, ..., Xk) 与响应 Y 的联合建模。
    计算各块的潜变量、加载矩阵、以及块重要性。

    参数
    ----
    n_components : int, 默认=5
        潜变量数量
    scale : bool, 默认=True
        是否对每块做标准差归一化
    block_names : list of str, 可选
        各块名称, 如 ["metabolome", "transcriptome", "proteome"]
    """

    def __init__(
        self,
        n_components: int = 5,
        scale: bool = True,
        block_names: Optional[List[str]] = None,
    ):
        self.n_components = n_components
        self.scale = scale
        self.block_names = block_names or []

        # 训练后填充
        self.n_blocks = 0
        self.block_shapes = []
        self.scalers: List[StandardScaler] = []
        self.block_pls: List[PLSRegression] = []
        self.block_scores: List[np.ndarray] = []     # 每块的得分 T_i
        self.block_loadings: List[np.ndarray] = []   # 每块的加载 P_i
        self.block_weights: List[np.ndarray] = []    # 每块的权重 W_i
        self.super_scores: Optional[np.ndarray] = None  # 超得分 T_T
        self.super_weights: Optional[np.ndarray] = None  # 超权重 w_T
        self.block_importance: Optional[np.ndarray] = None  # 块重要性
        self.vip_scores: List[np.ndarray] = []       # 各块 VIP 分数
        self.y_loadings: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        blocks: List[np.ndarray],
        y: np.ndarray,
    ) -> "MultiBlockPLS":
        """
        训练 MB-PLS 模型

        参数
        ----
        blocks : list of np.ndarray
            数据块列表 [X1, X2, ..., Xk], 每块 (n_samples, n_features)
        y : np.ndarray (n_samples,) 或 (n_samples, 1)
            响应变量

        返回
        ----
        self
        """
        self.n_blocks = len(blocks)
        n_samples = blocks[0].shape[0]

        # 校验
        for i, X in enumerate(blocks):
            assert X.shape[0] == n_samples, f"Block {i} has {X.shape[0]} samples, expected {n_samples}"
        assert y.shape[0] == n_samples, f"y has {y.shape[0]} samples, expected {n_samples}"
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # 设置块名
        if len(self.block_names) < self.n_blocks:
            self.block_names = [f"Block_{i+1}" for i in range(self.n_blocks)]

        # 标准化每块
        self.scalers = []
        X_scaled = []
        for i, X in enumerate(blocks):
            scl = StandardScaler(with_std=self.scale)
            X_s = scl.fit_transform(X)
            self.scalers.append(scl)
            X_scaled.append(X_s)

        self.block_shapes = [X.shape for X in blocks]

        # ---- Step 1: 对每块独立做 PLS, 提取得分 ----
        self.block_scores = []
        self.block_loadings = []
        self.block_weights = []
        self.block_pls = []

        for i, X_s in enumerate(X_scaled):
            pls = PLSRegression(n_components=self.n_components, scale=False)
            pls.fit(X_s, y)
            T_i = pls.x_scores_  # (n_samples, n_components)
            P_i = pls.x_loadings_  # (n_features, n_components)
            W_i = pls.x_weights_  # (n_features, n_components)
            self.block_scores.append(T_i)
            self.block_loadings.append(P_i)
            self.block_weights.append(W_i)
            self.block_pls.append(pls)

        # ---- Step 2: 超层 PLS —— 用各块得分拼起来对 y 做 PLS ----
        # 拼合所有块的得分矩阵
        T_concat = np.hstack(self.block_scores)  # (n, k * n_components)

        super_pls = PLSRegression(n_components=min(self.n_components, T_concat.shape[1]), scale=False)
        super_pls.fit(T_concat, y)
        self.super_scores = super_pls.x_scores_
        self.super_weights = super_pls.x_weights_  # (k*n_components, super_n_comp)
        self.y_loadings = super_pls.y_loadings_

        # ---- Step 3: 计算块重要性 ----
        self._compute_block_importance()

        # ---- Step 4: 计算各块的 VIP 分数 ----
        self._compute_vip_scores(y)

        self._fitted = True
        return self

    def _compute_block_importance(self):
        """基于超权重的块重要性评估"""
        k = self.n_blocks
        nc = self.n_components
        # super_weights shape: (k * n_components, super_n_components)
        # 每个块贡献 nc 行
        importance = np.zeros(k)
        for i in range(k):
            idx_start = i * nc
            idx_end = (i + 1) * nc
            if self.super_weights is not None:
                block_w = self.super_weights[idx_start:idx_end, :]  # (nc, super_nc)
                importance[i] = np.sum(np.abs(block_w))
        # 归一化
        total = np.sum(importance)
        self.block_importance = importance / total if total > 0 else importance

    def _compute_vip_scores(self, y: np.ndarray):
        """
        计算各块每个变量的 VIP (Variable Importance in Projection)

        VIP_j = sqrt( p * sum_a [w_{ja}^2 * SSY_a] / sum_a SSY_a )

        其中 p 是变量数, w_{ja} 是第 a 个潜变量的权重,
        SSY_a 是该潜变量解释的 Y 方差
        """
        for i in range(self.n_blocks):
            W = self.block_weights[i]  # (n_features, n_components)
            T = self.block_scores[i]  # (n_samples, n_components)
            n_features = W.shape[0]

            # 每个潜变量解释的 Y 方差 (简化为得分方差 * Y 相关性)
            ssy = np.zeros(self.n_components)
            for a in range(self.n_components):
                t_a = T[:, a]
                y_flat = y.flatten() if y.ndim > 1 else y
                corr = np.corrcoef(t_a, y_flat)[0, 1]
                ssy[a] = (corr ** 2) * np.sum(t_a ** 2)

            ssy_total = np.sum(ssy)
            if ssy_total == 0:
                ssy_total = 1e-10

            vip = np.zeros(n_features)
            for j in range(n_features):
                numerator = n_features * np.sum((W[j, :] ** 2) * ssy)
                vip[j] = np.sqrt(numerator / ssy_total)

            self.vip_scores.append(vip)

    def transform(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        将新数据投影到潜变量空间

        参数
        ----
        blocks : list of np.ndarray
            新数据块 [X1_new, X2_new, ...]

        返回
        ----
        T_super : np.ndarray (n_samples, n_components)
            超得分
        """
        assert self._fitted, "Model must be fitted first"
        X_scaled = []
        for i, X in enumerate(blocks):
            X_s = self.scalers[i].transform(X)
            X_scaled.append(X_s)

        # 各块得分
        T_list = []
        for i, X_s in enumerate(X_scaled):
            result = self.block_pls[i].transform(X_s, copy=True)
            if isinstance(result, tuple):
                T_i = result[0]
            else:
                T_i = result
            T_list.append(T_i)

        T_concat = np.hstack(T_list)
        return T_concat @ self._super_weights_normalized()

    def _super_weights_normalized(self) -> np.ndarray:
        """获取标准化超权重"""
        return self.super_weights

    def get_driving_metabolites(
        self,
        block_idx: int = 0,
        top_k: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        获取第 block_idx 块的驱动代谢物 (按 VIP 排序)

        返回
        ----
        dict: {"indices": array, "vip": array}
        """
        assert self._fitted, "Model must be fitted first"
        vip = self.vip_scores[block_idx]
        indices = np.argsort(vip)[::-1][:top_k]
        return {
            "block_name": self.block_names[block_idx],
            "indices": indices,
            "vip_values": vip[indices],
            "top_k": top_k,
        }

    def residual_space(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        计算各块的残差空间 (用于主动学习的 uncertainty sampling)

        X_residual = X - T @ P^T

        返回
        ----
        residuals : list of np.ndarray
            每块的残差矩阵
        """
        assert self._fitted, "Model must be fitted first"
        residuals = []
        for i, X in enumerate(blocks):
            X_s = self.scalers[i].transform(X)
            T_i = self.block_scores[i]
            P_i = self.block_loadings[i]
            X_reconstructed = T_i @ P_i.T
            residuals.append(X_s - X_reconstructed)
        return residuals

    def uncertainty_score(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        基于残差空间的样本不确定性分数

        u_i = ||r_i||_2  (每个样本在所有块上的残差 L2 范数之和)

        分数越高 → 模型对该样本的拟合越差 → 主动学习应优先选择

        返回
        ----
        uncertainty : np.ndarray (n_samples,)
        """
        residuals = self.residual_space(blocks)
        n_samples = blocks[0].shape[0]
        uncertainty = np.zeros(n_samples)
        for R in residuals:
            uncertainty += np.linalg.norm(R, axis=1)
        return uncertainty

    def summary(self) -> str:
        """打印模型摘要"""
        if not self._fitted:
            return "MultiBlockPLS (not fitted)"

        lines = [
            "=" * 60,
            "  Multi-Block PLS Model Summary",
            "=" * 60,
            f"  Number of blocks:       {self.n_blocks}",
            f"  Latent components:      {self.n_components}",
            f"  Samples:                {self.block_shapes[0][0]}",
        ]
        for i, (name, shape) in enumerate(zip(self.block_names, self.block_shapes)):
            lines.append(f"  {name}: {' ' * (20 - len(name))} {shape[1]} features")
        lines.append("-" * 60)
        lines.append("  Block Importance:")
        for i, (name, imp) in enumerate(zip(self.block_names, self.block_importance)):
            bar = "█" * int(imp * 40)
            lines.append(f"    {name:20s} {imp:.3f}  {bar}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出模型参数为字典 (可序列化)"""
        return {
            "n_components": self.n_components,
            "n_blocks": self.n_blocks,
            "block_names": self.block_names,
            "block_shapes": [(int(s[0]), int(s[1])) for s in self.block_shapes],
            "block_importance": self.block_importance.tolist() if self.block_importance is not None else None,
            "fitted": self._fitted,
        }


# ============================================================
#  便捷函数: 生成合成多组学数据 (用于轻薄本演示)
# ============================================================

def generate_toy_multiblock_data(
    n_samples: int = 100,
    n_metabolites: int = 50,
    n_transcripts: int = 200,
    n_proteins: int = 80,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    生成三块合成数据用于演示

    模拟:
      - 代谢组 X1: (n_samples, n_metabolites)
      - 转录组 X2: (n_samples, n_transcripts)
      - 蛋白组 X3: (n_samples, n_proteins)
      - 响应 Y: 生长速率 (受潜变量驱动)

    返回
    ----
    blocks : [X1, X2, X3]
    y : 响应
    feature_names : 各块特征名列表
    """
    rng = np.random.RandomState(seed)

    # 真实潜变量 (驱动三块 + Y 的共同信号)
    latent_true = rng.randn(n_samples, 3)  # 3 个底层因子

    # 各块的加载矩阵 (潜变量 → 观测特征)
    W1 = rng.randn(3, n_metabolites) * 0.3
    W2 = rng.randn(3, n_transcripts) * 0.2
    W3 = rng.randn(3, n_proteins) * 0.25

    # 生成观测
    X1 = latent_true @ W1 + noise * rng.randn(n_samples, n_metabolites)
    X2 = latent_true @ W2 + noise * rng.randn(n_samples, n_transcripts)
    X3 = latent_true @ W3 + noise * rng.randn(n_samples, n_proteins)

    # Y 也由潜变量驱动
    y = latent_true @ np.array([1.0, 0.8, 0.5]) + noise * rng.randn(n_samples)

    feature_names = [
        [f"met_{i}" for i in range(n_metabolites)],
        [f"gene_{i}" for i in range(n_transcripts)],
        [f"prot_{i}" for i in range(n_proteins)],
    ]

    return [X1, X2, X3], y, feature_names
