"""
Surrogate 模型 —— 虚拟湿实验代理
=======================================
用 Stage 1 训练好的 MB-PLS 模型作为 surrogate,
预测双敲除对代谢表型的影响。

然后在工作站上运行 FBA 验证,
减少真实湿实验次数。

功能:
  - PLS 潜变量 → 预测生长率
  - 对比预测 vs 实测 (FBA 结果)
  - 残差分析指导下一轮选样
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class SurrogateModel:
    """
    基于 MB-PLS 的 Surrogate 模型

    用 MB-PLS 潜变量预测双敲除后的生长率,
    作为真实 FBA/virtual-cell 实验的低成本代理。

    参数
    ----
    n_components : int
        用于预测的潜变量数
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.beta: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self._fitted = False
        self._train_residuals: Optional[np.ndarray] = None

    def fit(
        self,
        latent_scores: np.ndarray,
        growth_rates: np.ndarray,
    ):
        """
        用潜变量拟合生长率预测模型

        y = latent @ beta + intercept

        参数
        ----
        latent_scores : np.ndarray (n_train, n_components)
            MB-PLS 超得分
        growth_rates : np.ndarray (n_train,)
            对应样本的生长率 (来自 FBA 或观测)
        """
        X = latent_scores[:, : self.n_components]
        y = growth_rates.flatten()

        # 最小二乘
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        theta = np.linalg.lstsq(X_aug, y, rcond=None)[0]

        self.intercept = float(theta[0])
        self.beta = theta[1:]
        self._fitted = True
        self._train_residuals = y - self.predict(latent_scores)
        return self

    def predict(self, latent_scores: np.ndarray) -> np.ndarray:
        """预测生长率"""
        assert self._fitted, "模型未训练"
        X = latent_scores[:, : self.n_components]
        return X @ self.beta + self.intercept

    def predict_with_uncertainty(
        self,
        latent_scores: np.ndarray,
        n_bootstrap: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带 Bootstrap 不确定性的预测

        返回
        ----
        mean : np.ndarray
            预测均值
        lower : np.ndarray
            95% CI 下界
        upper : np.ndarray
            95% CI 上界
        """
        assert self._fitted, "模型未训练"

        n_samples = latent_scores.shape[0]
        bootstrap_preds = np.zeros((n_bootstrap, n_samples))

        # 残差 Bootstrap
        residuals = self._train_residuals.copy()
        for b in range(n_bootstrap):
            idx = np.random.randint(0, len(residuals), n_samples)
            y_boot = self.predict(latent_scores) + residuals[idx]
            bootstrap_preds[b] = y_boot

        mean = bootstrap_preds.mean(axis=0)
        std = bootstrap_preds.std(axis=0)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        return mean, lower, upper

    def evaluate(
        self,
        latent_scores: np.ndarray,
        true_growth: np.ndarray,
    ) -> Dict[str, float]:
        """
        评估 surrogate 模型性能

        返回
        ----
        metrics : dict
            {"r2": ..., "rmse": ..., "mae": ..., "spearman_r": ...}
        """
        from scipy.stats import spearmanr

        pred = self.predict(latent_scores)
        true = true_growth.flatten()

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        mae = np.mean(np.abs(true - pred))
        spear, _ = spearmanr(true, pred)

        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "spearman_r": float(spear),
            "n_samples": len(true),
        }

    def get_residuals(
        self, latent_scores: np.ndarray, true_growth: np.ndarray
    ) -> np.ndarray:
        """计算预测残差"""
        pred = self.predict(latent_scores)
        return true_growth.flatten() - pred

    @staticmethod
    def contrast_models(
        mbpls_predictions: np.ndarray,
        transcript_only_predictions: np.ndarray,
        true_growth: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        对比 MB-PLS vs 纯转录组的预测性能

        这是方法学文章的关键对比表格

        返回
        ----
        comparison : dict
            {"mbpls": {...metrics}, "transcript_only": {...metrics}}
        """
        from scipy.stats import spearmanr

        def compute_metrics(pred, true):
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            spear, _ = spearmanr(true, pred)
            return {"r2": r2, "rmse": rmse, "spearman_r": spear}

        return {
            "mbpls_chemometric": compute_metrics(mbpls_predictions, true_growth),
            "transcript_only_baseline": compute_metrics(
                transcript_only_predictions, true_growth
            ),
        }

    def summary(self) -> str:
        """模型摘要"""
        if not self._fitted:
            return "SurrogateModel (not fitted)"
        return (
            f"SurrogateModel(n_components={self.n_components}, "
            f"r2_train={1 - np.var(self._train_residuals):.3f})"
        )
