"""
实验设计器 (DoE) —— 化学计量学老本行
===========================================
将 DoE 方法嵌入主动学习闭环:

  - 中心复合设计 (CCD) 用于参数空间探索
  - 因子设计用于双敲除组合筛选
  - 最优设计 (D-optimal) 用于选样

这是衔接 "MB-PLS 残差" 与 "双敲除选样" 的桥梁。
"""

import numpy as np
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple


class ExperimentDesigner:
    """
    DoE 实验设计器

    参数
    ----
    n_factors : int
        因子数量 (如: 关键代谢物数量)
    design_type : str
        "factorial", "ccd", "d_optimal", "fractional_factorial"
    """

    def __init__(
        self,
        n_factors: int = 5,
        design_type: str = "factorial",
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.design_type = design_type
        self.rng = np.random.RandomState(random_state)

    def generate_design(
        self,
        n_center: int = 3,
        alpha: float = 1.5,
    ) -> np.ndarray:
        """
        生成实验设计矩阵

        参数
        ----
        n_center : int
            中心点重复数
        alpha : float
            CCD 的轴向距离

        返回
        ----
        design : np.ndarray (n_runs, n_factors)
            编码设计矩阵 (-1, 0, 1 或 CCD axial)
        """
        if self.design_type == "factorial":
            design = self._full_factorial(n_center)
        elif self.design_type == "ccd":
            design = self._central_composite_design(n_center, alpha)
        elif self.design_type == "bbd":
            design = self._box_behnken_design(n_center)
        elif self.design_type == "fractional_factorial":
            design = self._fractional_factorial()
        elif self.design_type == "d_optimal":
            design = self._d_optimal_candidates(n_center)
        else:
            raise ValueError(f"Unsupported design: {self.design_type}")

        return design

    def _full_factorial(self, n_center: int) -> np.ndarray:
        """2^k 全因子设计 + 中心点"""
        design = np.array(list(product([-1, 1], repeat=self.n_factors)))
        # 添加中心点
        centers = np.zeros((n_center, self.n_factors))
        return np.vstack([design, centers])

    def _fractional_factorial(self) -> np.ndarray:
        """2^(k-1) 部分因子设计"""
        # 生成全因子, 取一半
        full = np.array(list(product([-1, 1], repeat=self.n_factors)))
        n_half = len(full) // 2
        indices = self.rng.choice(len(full), n_half, replace=False)
        return full[indices]

    def _central_composite_design(self, n_center: int, alpha: float) -> np.ndarray:
        """中心复合设计 (CCD)"""
        # 因子部分
        factorial = np.array(list(product([-1, 1], repeat=self.n_factors)))
        # 轴向点
        axial = []
        for i in range(self.n_factors):
            for sign in [-1, 1]:
                point = np.zeros(self.n_factors)
                point[i] = sign * alpha
                axial.append(point)
        axial = np.array(axial)
        # 中心点
        centers = np.zeros((n_center, self.n_factors))
        return np.vstack([factorial, axial, centers])

    def _box_behnken_design(self, n_center: int) -> np.ndarray:
        """Box-Behnken design (BBD).

        For k factors, BBD generates (k*(k-1))*2 + n_center runs.
        Each run fixes (k-2) factors at 0 while pairing the remaining 2
        factors in a 2^2 factorial. Requires k >= 3.
        """
        if self.n_factors < 3:
            raise ValueError(f"BBD requires at least 3 factors, got {self.n_factors}")
        design = []
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                for vi, vj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    point = np.zeros(self.n_factors)
                    point[i] = vi
                    point[j] = vj
                    design.append(point)
        centers = np.zeros((n_center, self.n_factors))
        return np.vstack([np.array(design), centers])

    def to_dataframe(self, design: np.ndarray, factor_names=None) -> "pd.DataFrame":
        """Convert design matrix to labelled DataFrame."""
        import pandas as pd

        if factor_names is None:
            factor_names = [f"X{i+1}" for i in range(design.shape[1])]
        return pd.DataFrame(design, columns=factor_names)

    def _d_optimal_candidates(self, n_center: int) -> np.ndarray:
        """
        D-最优设计的近似实现
        生成大量候选点, 用贪心算法选 D-最优子集
        """
        # 生成候选集 (3 水平)
        candidates = np.array(list(product([-1, 0, 1], repeat=self.n_factors)))
        n_candidates = len(candidates)

        # 贪心选 D-最优: 每次加入使 det(X'X) 最大的点
        n_select = min(2 * self.n_factors + n_center, n_candidates)
        selected = [candidates[0]]

        for _ in range(n_select - 1):
            best_det = -1
            best_idx = -1
            for i in range(n_candidates):
                if i in [
                    np.where((candidates == s).all(axis=1))[0][0] for s in selected
                ]:
                    continue
                trial = np.vstack(selected + [candidates[i]])
                det = np.linalg.det(trial.T @ trial)
                if det > best_det:
                    best_det = det
                    best_idx = i
            if best_idx >= 0:
                selected.append(candidates[best_idx])

        return np.array(selected)

    def design_knockout_pairs(
        self,
        gene_pool: List[str],
        factor_indices: List[int],
        max_pairs: int = 200,
    ) -> List[Tuple[str, str]]:
        """
        基于 DoE 设计双敲除对

        将实验设计的因子水平映射到具体基因:
          因子 i 的 -1 水平 → 敲除基因 A_i
          因子 i 的 +1 水平 → 敲除基因 B_i

        参数
        ----
        gene_pool : list of str
            候选基因列表
        factor_indices : list of int
            选作因子的基因索引 (通常来自 MB-PLS VIP 前 N)
        max_pairs : int
            最大配对数

        返回
        ----
        gene_pairs : list of (str, str)
        """
        design = self.generate_design()
        pairs = []

        for row in design:
            for i in range(self.n_factors - 1):
                for j in range(i + 1, self.n_factors):
                    if i < len(factor_indices) and j < len(factor_indices):
                        ga = (
                            gene_pool[factor_indices[i]]
                            if factor_indices[i] < len(gene_pool)
                            else f"G{factor_indices[i]}"
                        )
                        gb = (
                            gene_pool[factor_indices[j]]
                            if factor_indices[j] < len(gene_pool)
                            else f"G{factor_indices[j]}"
                        )
                        # 按水平决定是否敲除
                        if row[i] != 0 and row[j] != 0:
                            pairs.append((ga, gb))

        # 去重
        pairs = list(set(pairs))
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        return pairs

    def latin_hypercube(
        self,
        n_samples: int = 50,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        拉丁超立方采样 —— 用于虚拟实验的参数空间探索

        参数
        ----
        n_samples : int
            采样点数
        bounds : list of (low, high), optional
            每维的边界, 默认 [0, 1]

        返回
        ----
        samples : np.ndarray (n_samples, n_factors)
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * self.n_factors

        samples = np.zeros((n_samples, self.n_factors))
        for j in range(self.n_factors):
            low, high = bounds[j]
            # 每维均匀分割区间
            intervals = np.linspace(low, high, n_samples + 1)
            # 每个区间内随机采样
            for i in range(n_samples):
                samples[i, j] = self.rng.uniform(intervals[i], intervals[i + 1])

        # 每维随机排列
        for j in range(self.n_factors):
            self.rng.shuffle(samples[:, j])

        return samples

    def summary(self) -> str:
        """设计信息"""
        return (
            f"ExperimentDesigner(type={self.design_type}, "
            f"n_factors={self.n_factors})"
        )
