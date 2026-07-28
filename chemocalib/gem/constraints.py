"""
潜变量 → GEM 约束映射器
================================
将 MB-PLS/DIABLO 的潜变量转换为 COBRApy 代谢网络约束。

核心概念:
  - 潜变量得分 → 代谢物丰度先验 → density_constraint
  - VIP 筛选的驱动代谢物 → 针对性反应约束
  - 多靶点约束: 同时约束多个反应的上/下界

这是 "计量学校准的约束 FBA" 的核心实现。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


class LatentToConstraint:
    """
    将 PLS 潜变量映射到 GEM 反应约束

    映射策略:
      1. 潜变量 → metabolite abundance proxy (softmax 归一化)
      2. 驱动代谢物 (VIP top-K) → 对应交换反应/转运反应的 bounds
      3. 支持 additive / multiplicative 两种约束模式

    参数
    ----
    metabolite_map : dict
        代谢物特征名 → GEM 代谢物 ID 映射, 如 {"met_0": "glc__D_c", ...}
    scaling_mode : str, "soft" | "hard" | "adaptive"
        - soft: 潜变量作为密度惩罚项 (density constraint)
        - hard: 直接修改反应上下界
        - adaptive: 根据置信度自适应调整
    """

    def __init__(
        self,
        metabolite_map: Optional[Dict[str, str]] = None,
        scaling_mode: str = "soft",
        min_bound: float = 0.0,
        max_bound: float = 1000.0,
        default_flux: float = 10.0,
    ):
        self.metabolite_map = metabolite_map or {}
        self.scaling_mode = scaling_mode
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.default_flux = default_flux

        # 缓存
        self._last_constraints: Optional[Dict] = None
        self._feature_to_reaction: Optional[Dict[int, str]] = None

    def build_feature_reaction_map(
        self,
        feature_names: List[str],
        gem_metabolite_ids: List[str],
        vip_scores: Optional[np.ndarray] = None,
        top_k: int = 30,
    ) -> Dict[int, str]:
        """
        构建 PLS 特征 → GEM 反应的映射表

        使用 VIP 分数选择关键代谢物, 映射到对应交换反应。
        命名约定: 代谢物 "met_X" → 交换反应 "EX_met_X"

        参数
        ----
        feature_names : list
            PLS 特征名 (如 met_0, met_1, ...)
        gem_metabolite_ids : list
            GEM 中的代谢物 ID 列表
        vip_scores : np.ndarray, 可选
            VIP 分数, 用于选 Top-K
        top_k : int
            选取的关键代谢物数量

        返回
        ----
        feature_to_rxn : dict
            {PLS特征索引: GEM反应ID}
        """
        if vip_scores is not None and len(vip_scores) > 0:
            top_indices = np.argsort(vip_scores)[::-1][:top_k]
        else:
            top_indices = np.arange(min(top_k, len(feature_names)))

        feature_to_rxn = {}
        for i, feat_idx in enumerate(top_indices):
            feat_name = (
                feature_names[feat_idx]
                if feat_idx < len(feature_names)
                else f"met_{feat_idx}"
            )
            # 尝试从映射表查找
            if feat_name in self.metabolite_map:
                rxn_id = self.metabolite_map[feat_name]
            else:
                # 默认命名: EX_ + 代谢物 ID
                if i < len(gem_metabolite_ids):
                    rxn_id = f"EX_{gem_metabolite_ids[i]}"
                else:
                    rxn_id = f"EX_met_{feat_idx}"
            feature_to_rxn[feat_idx] = rxn_id

        self._feature_to_reaction = feature_to_rxn
        return feature_to_rxn

    def latent_to_bounds(
        self,
        latent_scores: np.ndarray,
        n_components: int = 3,
        scale_factor: float = 100.0,
    ) -> Dict[str, Tuple[float, float]]:
        """
        将潜变量得分转为反应上下界

        算法:
          对每个潜变量分量, 计算其在样本间的分布,
          映射到 [min_bound, max_bound] 区间作为反应通量边界

        参数
        ----
        latent_scores : np.ndarray (n_components,)
            单个或平均潜变量得分
        n_components : int
            使用的潜变量分量数
        scale_factor : float
            缩放因子 (潜变量标准差 → 通量量纲)

        返回
        ----
        bounds : dict {rxn_id: (lb, ub)}
        """
        if self._feature_to_reaction is None:
            raise ValueError("请先调用 build_feature_reaction_map() 建立映射")

        bounds = {}
        latent_scores = np.asarray(latent_scores).flatten()
        latent = latent_scores[:n_components]

        for i, lv in enumerate(latent):
            if i >= len(self._feature_to_reaction):
                break
            feat_idx = list(self._feature_to_reaction.keys())[i]
            rxn_id = self._feature_to_reaction[feat_idx]

            if self.scaling_mode == "soft":
                # soft: 潜变量正值 → 增大通量上限, 负值 → 减小
                ub = self.default_flux * (
                    1.0 + np.clip(lv * scale_factor / 100.0, -0.9, 10.0)
                )
                lb = (
                    self.default_flux
                    * (1.0 - np.clip(lv * scale_factor / 100.0, -0.9, 10.0))
                    * (-1.0)
                )
                bounds[rxn_id] = (max(self.min_bound, lb), min(self.max_bound, ub))
            elif self.scaling_mode == "hard":
                # hard: 直接映射
                ub = np.clip(
                    self.default_flux + lv * scale_factor,
                    self.min_bound,
                    self.max_bound,
                )
                lb = np.clip(
                    -self.default_flux + lv * scale_factor,
                    self.min_bound,
                    self.max_bound,
                )
                if lb > ub:
                    lb, ub = ub, lb
                bounds[rxn_id] = (lb, ub)
            else:  # adaptive
                # Adaptive: larger |latent| → tighter constraint.
                # Positive latent → increase upper bound; negative → tighten lower bound.
                conf = 1.0 / (1.0 + np.exp(-abs(lv)))  # sigmoid confidence
                modulation = abs(conf * lv * scale_factor / 100.0)
                sign = 1 if lv >= 0 else -1
                if sign > 0:
                    ub = self.default_flux * (1.0 + modulation)
                    lb = -self.default_flux
                else:
                    ub = self.default_flux
                    lb = -self.default_flux * (1.0 + modulation)
                bounds[rxn_id] = (max(self.min_bound, lb), min(self.max_bound, ub))

        self._last_constraints = bounds
        return bounds

    def batch_constraints(
        self,
        latent_matrix: np.ndarray,
        n_components: int = 3,
    ) -> pd.DataFrame:
        """
        批量生成约束矩阵 (每行一个样本)

        参数
        ----
        latent_matrix : np.ndarray (n_samples, n_components)
            潜变量矩阵 (如 MB-PLS super_scores)
        n_components : int
            使用的分量数

        返回
        ----
        constraint_df : pd.DataFrame
            每行一个样本, 每列一个反应约束值 (上界)
            可与 COBRApy 批量接口对接
        """
        if self._feature_to_reaction is None:
            raise ValueError("请先调用 build_feature_reaction_map()")

        n_samples = latent_matrix.shape[0]
        rxn_ids = list(self._feature_to_reaction.values())
        n_rxns = min(n_components, len(rxn_ids))
        rxn_ids = rxn_ids[:n_rxns]

        bounds_matrix = np.zeros((n_samples, n_rxns))
        for i in range(n_samples):
            b = self.latent_to_bounds(
                latent_matrix[i, :],
                n_components=n_rxns,
            )
            for j, rxn in enumerate(rxn_ids):
                if rxn in b:
                    bounds_matrix[i, j] = b[rxn][1]  # 取上界

        return pd.DataFrame(bounds_matrix, columns=rxn_ids)

    def create_density_objective(
        self,
        latent_scores: np.ndarray,
        metabolite_pool: Dict[str, float],
        alpha: float = 0.1,
    ) -> Dict[str, float]:
        """
        创建代谢物密度目标函数

        用于 COBRApy 的 add_concentration_objective 或
        自定义目标: max c^T v - alpha * ||latent - observed||

        参数
        ----
        latent_scores : np.ndarray
            潜变量得分
        metabolite_pool : dict
            观测代谢物浓度 {met_id: concentration}
        alpha : float
            正则化强度

        返回
        ----
        objective_weights : dict
            {rxn_id: weight in objective}
        """
        weights = {}
        latent_flat = latent_scores.flatten()

        for i, (met_id, conc) in enumerate(metabolite_pool.items()):
            if i < len(latent_flat):
                # 潜变量接近观测 → 高权重
                weight = alpha * np.exp(
                    -abs(latent_flat[i] - conc) / (abs(conc) + 1e-8)
                )
                rxn_id = f"EX_{met_id}"
                weights[rxn_id] = float(weight)

        return weights

    def to_dataframe(self) -> pd.DataFrame:
        """将当前约束导出为 DataFrame"""
        if self._last_constraints is None:
            return pd.DataFrame()
        rows = []
        for rxn, (lb, ub) in self._last_constraints.items():
            rows.append({"reaction": rxn, "lower_bound": lb, "upper_bound": ub})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """约束摘要"""
        mode_labels = {
            "soft": "Soft density constraint",
            "hard": "Hard bound modification",
            "adaptive": "Adaptive confidence-weighted",
        }
        n = len(self._feature_to_reaction) if self._feature_to_reaction else 0
        return (
            f"LatentToConstraint(mode={self.scaling_mode} [{mode_labels.get(self.scaling_mode, '?')}], "
            f"mapped_reactions={n})"
        )
