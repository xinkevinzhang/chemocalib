"""
DIABLO 风格多块对齐 (MultiBlockAligner)
===========================================

实现 mixOmics DIABLO 框架的 Python 轻量版本:
  - 稀疏 PLS-DA 多块对齐
  - 跨块相关性最大化
  - 块间协方差结构分解

用于: 代谢组 + 转录组 + 蛋白组 三块对齐,
      提取共享/特异的潜变量结构

参考文献:
  - Singh et al. (2019) Bioinformatics (DIABLO)
  - Tenenhaus & Tenenhaus (2011) CSDA (RGCCA)
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
from scipy.linalg import svd


class MultiBlockAligner:
    """
    多块数据对齐器 (DIABLO-inspired)

    实现思路:
      1. 对每块独立做 PLS (或原始 PLS) 提取得分
      2. 在得分空间做联合 SVD 找到共享结构
      3. 选稀疏变量 (按 loading 大小)
      4. 评估块间相关性

    参数
    ----
    n_components : int
        潜变量数
    keep_sparse : float or list of float
        每块保留的变量比例 (默认 0.3)
    """

    def __init__(
        self,
        n_components: int = 5,
        keep_sparse: float = 0.3,
        block_names: Optional[List[str]] = None,
    ):
        self.n_components = n_components
        self.keep_sparse = keep_sparse
        self.block_names = block_names or []
        self._fitted = False

        self.scalers = []
        self.pls_models: List[PLSRegression] = []
        self.scores: List[np.ndarray] = []
        self.loadings: List[np.ndarray] = []
        self.shared_structure: Optional[np.ndarray] = None
        self.block_correlation: Optional[np.ndarray] = None
        self.selected_vars: List[np.ndarray] = []  # 每块选中的变量索引

    def fit(
        self,
        blocks: List[np.ndarray],
        y: np.ndarray,
    ) -> "MultiBlockAligner":
        """
        训练多块对齐模型

        参数
        ----
        blocks : [X1, X2, ..., Xk]
        y : 响应 (可用于有监督对齐)
        """
        self.n_blocks = len(blocks)
        n_samples = blocks[0].shape[0]
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        if len(self.block_names) < self.n_blocks:
            self.block_names = [f"Block_{i+1}" for i in range(self.n_blocks)]

        # 标准化
        self.scalers = []
        X_scaled = []
        for X in blocks:
            scl = StandardScaler()
            X_s = scl.fit_transform(X)
            self.scalers.append(scl)
            X_scaled.append(X_s)

        # 每块独立 PLS
        self.scores = []
        self.loadings = []
        self.pls_models = []

        for X_s in X_scaled:
            pls = PLSRegression(n_components=self.n_components, scale=False)
            pls.fit(X_s, y)
            self.pls_models.append(pls)
            self.scores.append(pls.x_scores_)
            self.loadings.append(pls.x_loadings_)

        # 联合 SVD: 找跨块共享结构
        # 把各块得分拼在一起做 SVD
        scores_concat = np.hstack(self.scores)  # (n, k * n_comp)
        # 对得分协方差矩阵做 SVD
        cov_scores = scores_concat.T @ scores_concat
        U_cov, s_cov, _ = svd(cov_scores)
        self.shared_structure = U_cov[:, : self.n_components]
        self.singular_values = s_cov

        # 块间相关性
        self.block_correlation = self._compute_block_correlation()

        # 稀疏变量选择
        self._select_sparse_variables(blocks)

        self._fitted = True
        return self

    def _compute_block_correlation(self) -> np.ndarray:
        """计算块间得分相关性矩阵"""
        k = self.n_blocks
        corr = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                # 取第一主成分的相关性
                c = np.corrcoef(self.scores[i][:, 0], self.scores[j][:, 0])[0, 1]
                corr[i, j] = abs(c)
                corr[j, i] = abs(c)
        return corr

    def _select_sparse_variables(self, blocks: List[np.ndarray]):
        """按 loading 绝对值选择稀疏变量"""
        if isinstance(self.keep_sparse, float):
            keep_list = [self.keep_sparse] * self.n_blocks
        else:
            keep_list = self.keep_sparse

        self.selected_vars = []
        for i, (L, X) in enumerate(zip(self.loadings, blocks)):
            # loading 重要性 = 跨所有潜变量的绝对值平均
            importance = np.mean(np.abs(L), axis=1)
            n_select = max(1, int(len(importance) * keep_list[i]))
            indices = np.argsort(importance)[::-1][:n_select]
            self.selected_vars.append(indices)

    def align_scores(self, blocks: List[np.ndarray]) -> np.ndarray:
        """
        对齐多块数据的得分

        返回
        ----
        aligned : np.ndarray (n_samples, n_components)
            对齐后的联合潜变量
        """
        assert self._fitted, "Model must be fitted first"
        X_scaled = [self.scalers[i].transform(X) for i, X in enumerate(blocks)]
        scores = []
        for i, X_s in enumerate(X_scaled):
            result = self.pls_models[i].transform(X_s, copy=True)
            if isinstance(result, tuple):
                T = result[0]
            else:
                T = result
            scores.append(T)
        scores_concat = np.hstack(scores)
        return scores_concat @ self.shared_structure[:, : self.n_components]

    def summary(self) -> str:
        """打印对齐摘要"""
        if not self._fitted:
            return "MultiBlockAligner (not fitted)"
        lines = ["=" * 60, "  DIABLO-style Multi-Block Alignment", "=" * 60]
        lines.append(f"  Blocks: {self.n_blocks}  |  Components: {self.n_components}")
        lines.append(f"  Singular values: {self.singular_values[:5].round(2)}")
        lines.append("\n  Block Correlation Matrix:")
        for i in range(self.n_blocks):
            row = " ".join(
                f"{self.block_correlation[i, j]:.3f}" for j in range(self.n_blocks)
            )
            lines.append(f"    {row}")
        lines.append("\n  Selected Sparse Variables:")
        for i, (name, idx) in enumerate(zip(self.block_names, self.selected_vars)):
            lines.append(f"    {name}: {len(idx)} / selected")
        lines.append("=" * 60)
        return "\n".join(lines)
