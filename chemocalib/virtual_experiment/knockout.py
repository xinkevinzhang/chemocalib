"""
虚拟双敲除实验 —— in-silico 双敲除生成器
=============================================
阶段二核心: 以 Stage 1 模型为 surrogate,
生成 200-500 个 in-silico 双敲除,
再用主动学习挑 10 个给合作者真做 (干湿闭环)。

功能:
  - 组合双敲除设计 (全组合 / DoE 筛选)
  - 批量 FBA 模拟
  - 表型 (生长率) 变化汇总
  - 合成致死 / 表型 epistasis 检测
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import warnings


class DoubleKnockoutDesigner:
    """
    双敲除实验设计器

    参数
    ----
    gene_pool : list of str
        候选基因列表
    n_min : int
        最小基因索引 (排除必需基因)
    design_strategy : str
        "exhaustive": 全组合
        "doe_filtered": DoE 筛选
        "vip_top": VIP 前 N 的组合
    """

    def __init__(
        self,
        gene_pool: Optional[List[str]] = None,
        design_strategy: str = "exhaustive",
        random_state: int = 42,
    ):
        self.gene_pool = gene_pool or []
        self.design_strategy = design_strategy
        self.rng = np.random.RandomState(random_state)

        self.pairs: List[Tuple[str, str]] = []
        self.results: Optional[pd.DataFrame] = None

    def generate_pairs(
        self,
        n_pairs: int = 200,
        vip_genes: Optional[List[str]] = None,
        exclude_genes: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """
        生成双敲除基因对

        参数
        ----
        n_pairs : int
            目标配对数 (200-500 为推荐范围)
        vip_genes : list of str, optional
            VIP 筛选出的高重要性基因 (优先组合)
        exclude_genes : list of str, optional
            排除基因 (如必需基因)
        """
        if not self.gene_pool and not vip_genes:
            raise ValueError("需要提供 gene_pool 或 vip_genes")

        exclude = set(exclude_genes) if exclude_genes else set()

        if self.design_strategy == "exhaustive":
            # 全组合 (受 n_pairs 上限控制)
            pool = vip_genes if vip_genes else self.gene_pool
            pool = [g for g in pool if g not in exclude]
            all_pairs = list(combinations(pool, 2))
            if len(all_pairs) > n_pairs:
                indices = self.rng.choice(len(all_pairs), n_pairs, replace=False)
                self.pairs = [all_pairs[i] for i in indices]
            else:
                self.pairs = all_pairs

        elif self.design_strategy == "vip_top":
            # VIP 前 N 之间交叉组合, 其余随机
            if not vip_genes:
                raise ValueError("vip_top 策略需要 vip_genes")
            vip_clean = [g for g in vip_genes if g not in exclude]
            other = [
                g
                for g in (self.gene_pool or [])
                if g not in vip_clean and g not in exclude
            ]

            # VIP × VIP 组合
            vip_pairs = list(combinations(vip_clean, 2))
            # VIP × other 组合
            cross_pairs = [(v, o) for v in vip_clean[:10] for o in other[:10]]
            # 随机 other × other
            other_pairs = list(combinations(other[:20], 2))

            all_pairs = vip_pairs + cross_pairs + other_pairs
            if len(all_pairs) > n_pairs:
                indices = self.rng.choice(len(all_pairs), n_pairs, replace=False)
                self.pairs = [all_pairs[i] for i in indices]
            else:
                self.pairs = all_pairs

        elif self.design_strategy == "doe_filtered":
            # 使用 DoE 设计矩阵生成对
            from chemocalib.active_learning.doe import ExperimentDesigner

            pool = [g for g in (self.gene_pool or []) if g not in exclude]
            doe = ExperimentDesigner(
                n_factors=min(n_pairs // 4, len(pool) // 2),
                design_type="factorial",
            )
            self.pairs = doe.design_knockout_pairs(
                pool, list(range(len(pool))), n_pairs
            )

        print(
            f"[DKO] 已生成 {len(self.pairs)} 组双敲除对 (策略: {self.design_strategy})"
        )
        return self.pairs

    def run_virtual_experiments(
        self,
        fba_simulator,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        运行虚拟双敲除实验 (批量 FBA)

        参数
        ----
        fba_simulator : FBASimulator
            已加载模型的 FBA 模拟器
        verbose : bool

        返回
        ----
        results : pd.DataFrame
        """
        if not self.pairs:
            raise ValueError("请先调用 generate_pairs() 生成基因对")

        results = fba_simulator.double_knockout_scan(
            gene_pairs=self.pairs,
            verbose=verbose,
        )
        self.results = results
        return results

    def analyze_results(self) -> Dict:
        """
        分析虚拟实验结果

        返回
        ----
        analysis : dict
            {
                "n_total": int,
                "n_lethal": int,
                "n_synthetic_lethal": int,  # 合成致死
                "n_suppressive": int,        # 抑制性 epistasis
                "growth_distribution": dict,
            }
        """
        if self.results is None or len(self.results) == 0:
            return {"error": "无结果, 请先运行 run_virtual_experiments()"}

        df = self.results
        growth = df["growth_double"].dropna().values

        analysis = {
            "n_total": len(df),
            "n_lethal": int(np.sum(df["is_lethal"])),
            "n_nonlethal": int(np.sum(~df["is_lethal"])),
            "growth_stats": {
                "mean": float(np.mean(growth)) if len(growth) > 0 else 0,
                "std": float(np.std(growth)) if len(growth) > 0 else 0,
                "min": float(np.min(growth)) if len(growth) > 0 else 0,
                "max": float(np.max(growth)) if len(growth) > 0 else 0,
                "median": float(np.median(growth)) if len(growth) > 0 else 0,
            },
        }

        # 基因参与度的度分布
        gene_counts = {}
        for _, row in df.iterrows():
            for g in [row["gene_a"], row["gene_b"]]:
                gene_counts[g] = gene_counts.get(g, 0) + 1
        # Top 10 most appeared genes
        top_genes = sorted(gene_counts.items(), key=lambda x: -x[1])[:10]
        analysis["top_genes_by_participation"] = top_genes

        return analysis

    def export_results(self, path: str):
        """导出结果到 CSV"""
        if self.results is not None:
            self.results.to_csv(path, index=False)
            print(f"[DKO] 结果已导出到 {path}")

    def generate_single_knockouts(self) -> List[str]:
        """Generate list of single-gene knockout targets.

        Returns
        -------
        list of str
            All genes in the pool as single-KO candidates.
        """
        return list(self.gene_pool)

    def summary(self) -> str:
        """设计器摘要"""
        return (
            f"DoubleKnockoutDesigner(strategy={self.design_strategy}, "
            f"pools={len(self.gene_pool)}, pairs={len(self.pairs)})"
        )
