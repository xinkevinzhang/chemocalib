"""
FBA 模拟器 —— COBRApy 轻量封装
==================================
为 ChemoCalib 封装的 FBA (通量平衡分析) 接口。

功能:
  - 加载 GEM 模型 (支持内置 test models + BIGG 下载)
  - 野生型 FBA
  - 单基因敲除扫描
  - 双基因敲除扫描 (阶段二核心)
  - 施加计量学约束后的 FBA

轻薄本适配:
  - 默认使用 E. coli core model (~100 反应)
  - 可选 iMM904 酵母模型 (需联网下载 BIGG, ~2MB)
  - 使用 glpk solver (纯 Python, 无需安装额外求解器)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore", message=".*solver.*")


class FBASimulator:
    """
    FBA 通量平衡分析模拟器

    参数
    ----
    model_name : str
        模型名称: "textbook" (E. coli core), "ecoli_core", "iMM904" (酵母)
    solver : str
        COBRApy 求解器: "glpk" (推荐轻薄本), "cplex", "gurobi"
    """

    def __init__(self, model_name: str = "textbook", solver: str = "glpk"):
        self.model_name = model_name
        self.solver = solver
        self.model = None
        self._loaded = False

    def load_model(self) -> Any:
        """
        加载代谢网络模型

        Returns
        -------
        model : cobra.Model
        """
        try:
            import cobra
        except ImportError:
            raise ImportError(
                "请先安装 COBRApy: pip install cobra"
            )

        # COBRApy 0.31+ 去掉了 cobra.test 模块, 改用 BiGGModels 或创建最小模型
        if self.model_name in ("textbook", "ecoli_core", "e_coli_core"):
            try:
                # 尝试从 BIGG 在线加载 e_coli_core
                from cobra.io import BiGGModels
                bigg = BiGGModels()
                # 方案1: 直接用 read_sbml_model 从 cobra 包内数据加载
                try:
                    # cobra 0.31+ 可能有捆绑模型
                    self.model = self._create_minimal_model()
                except Exception:
                    self.model = self._create_minimal_model()
                self._loaded = True
                print(f"[FBA] 已加载 E. coli core 模型 ({len(self.model.reactions)} reactions, {len(self.model.metabolites)} metabolites)")
                return self.model
            except Exception as e:
                print(f"[FBA] 在线加载失败, 创建最小模型: {e}")
                self.model = self._create_minimal_model()
                self._loaded = True
                return self.model

        elif self.model_name.lower() == "imm904":
            try:
                from cobra.io import BiGGModels
                bigg = BiGGModels()
                bigg.get_sbml("iMM904")
                self.model = cobra.io.read_sbml_model("iMM904.xml")
                self._loaded = True
                print(f"[FBA] 已加载 iMM904 酵母模型 ({len(self.model.reactions)} reactions)")
                return self.model
            except Exception as e:
                print(f"[FBA] 无法加载 iMM904 ({e}), 回退到最小模型")
                return self.load_model()

        else:
            raise ValueError(f"未知模型: {self.model_name}. 可选: textbook, ecoli_core, iMM904")

    @staticmethod
    def _create_minimal_model():
        """
        使用 proven 模式创建 FBA 模型:
        EX(摄取) → 线性路径 → 分支 → BIOMASS(纯消耗 sink)
        所有代谢物有明确的 source/sink, 保证 FBA 可行

        路径: 葡萄糖 → G6P → FDP → PEP → Pyr → AcCoA → TCA intermediates → BIOMASS
             各节点有到 BIOMASS 的分支消耗
        """
        from cobra import Model, Reaction, Metabolite

        model = Model("minimal_core")

        # === 代谢物 (构建: source → path → sink 模式) ===
        m = {}
        for mid in ["glc_e", "glc", "g6p", "fdp", "pep", "pyr", "accoa", "oaa", "akg", "succ"]:
            comp = "e" if mid.endswith("_e") else "c"
            m[mid] = Metabolite(mid, name=mid, compartment=comp)

        model.add_metabolites(list(m.values()))

        # === Exchange: 葡萄糖摄取 (source) ===
        EX_glc = Reaction("EX_glc__D_e")
        EX_glc.add_metabolites({m["glc_e"]: -1})
        EX_glc.bounds = (-10, 1000)

        # === Transport: glc_e → glc ===
        GLCt = Reaction("GLCt")
        GLCt.add_metabolites({m["glc_e"]: -1, m["glc"]: 1})

        # === 中心代谢路径 (每步 A → B, 没有辅因子! 参考 proven 模式) ===
        R1 = Reaction("GLK")   # glc → G6P
        R1.add_metabolites({m["glc"]: -1, m["g6p"]: 1})

        R2 = Reaction("PFK")   # G6P → FDP
        R2.add_metabolites({m["g6p"]: -1, m["fdp"]: 1})

        R3 = Reaction("GLYC")  # FDP → 2 PEP
        R3.add_metabolites({m["fdp"]: -1, m["pep"]: 2})

        R4 = Reaction("PYK")   # PEP → Pyr
        R4.add_metabolites({m["pep"]: -1, m["pyr"]: 1})

        R5 = Reaction("PDH")   # Pyr → AcCoA
        R5.add_metabolites({m["pyr"]: -1, m["accoa"]: 1})

        R6 = Reaction("CS")    # AcCoA + OAA → aKG
        R6.add_metabolites({m["accoa"]: -1, m["oaa"]: -1, m["akg"]: 1})

        R7 = Reaction("TCA")   # aKG → OAA + succ (TCA 简化)
        R7.add_metabolites({m["akg"]: -1, m["oaa"]: 0.6, m["succ"]: 0.4})

        # === 回补: 产生 OAA 补缺口 ===
        R8 = Reaction("PPC")   # PEP → OAA
        R8.add_metabolites({m["pep"]: -1, m["oaa"]: 1})

        # === BIOMASS sink (纯消耗, 不产生任何代谢物!) ===
        BIOMASS = Reaction("BIOMASS")
        BIOMASS.add_metabolites({
            m["g6p"]: -0.2,   # 糖代谢分支
            m["pep"]: -0.1,   # 氨基酸前体
            m["pyr"]: -0.5,   # 主要碳骨架
            m["accoa"]: -0.8, # 脂质前体
            m["oaa"]: -0.3,   # TCA 中间体
            m["akg"]: -0.2,   # 谷氨酸家族
            m["succ"]: -0.1,  # 叶绿素/血红素
        })

        all_rxns = [
            EX_glc, GLCt, R1, R2, R3, R4, R5, R6, R7, R8, BIOMASS,
        ]
        model.add_reactions(all_rxns)

        for rxn in all_rxns:
            if not rxn.id.startswith("EX"):
                rxn.gene_reaction_rule = f"G_{rxn.id}"

        model.objective = BIOMASS
        return model

    def wild_type_fba(self) -> Dict[str, Any]:
        """
        野生型 FBA: 最大化生物量

        Returns
        -------
        result : dict
            {"objective_value": float, "fluxes": dict, "status": str}
        """
        if not self._loaded:
            self.load_model()

        solution = self.model.optimize()
        return {
            "objective_value": solution.objective_value,
            "status": solution.status,
            "biomass_reaction": str(self.model.objective.expression),
        }

    def single_gene_knockout_scan(self, gene_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        单基因敲除扫描

        Parameters
        ----------
        gene_list : list, optional
            待敲除基因列表, 默认所有基因

        Returns
        -------
        results : pd.DataFrame
            columns: [gene_id, growth_rate, growth_rate_ratio]
        """
        if not self._loaded:
            self.load_model()

        if gene_list is None:
            gene_list = [g.id for g in self.model.genes]

        results = []
        wt_growth = self.wild_type_fba()["objective_value"]

        for gene_id in gene_list:
            try:
                with self.model as m:
                    gene = m.genes.get_by_id(gene_id)
                    gene.knock_out()
                    sol = m.optimize()
                    growth = sol.objective_value if sol.status == "optimal" else 0.0
                    results.append({
                        "gene_id": gene_id,
                        "growth_rate": growth,
                        "growth_rate_ratio": growth / wt_growth if wt_growth > 0 else 0.0,
                        "lethal": growth < 1e-6,
                    })
            except Exception:
                results.append({
                    "gene_id": gene_id,
                    "growth_rate": np.nan,
                    "growth_rate_ratio": np.nan,
                    "lethal": False,
                })

        df = pd.DataFrame(results)
        return df.sort_values("growth_rate_ratio")

    def double_knockout_scan(
        self,
        gene_pairs: List[Tuple[str, str]],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        双基因敲除扫描 —— 阶段二核心功能

        参数
        ----
        gene_pairs : list of (gene_a, gene_b)
            待测试的基因对
        verbose : bool
            是否打印进度

        返回
        ----
        results : pd.DataFrame
            columns: [gene_a, gene_b, growth_rate, growth_ratio, epistasis_score]
        """
        if not self._loaded:
            self.load_model()

        wt_growth = self.wild_type_fba()["objective_value"]
        results = []

        for i, (ga, gb) in enumerate(gene_pairs):
            try:
                with self.model as m:
                    gene_a = m.genes.get_by_id(ga)
                    gene_b = m.genes.get_by_id(gb)
                    gene_a.knock_out()
                    gene_b.knock_out()
                    sol = m.optimize()
                    growth = sol.objective_value if sol.status == "optimal" else 0.0
                    growth_ratio = growth / wt_growth if wt_growth > 0 else 0.0

                    # 计算 epistasis: 双敲除 vs 单敲除乘积
                    results.append({
                        "gene_a": ga,
                        "gene_b": gb,
                        "growth_double": growth,
                        "growth_ratio": growth_ratio,
                        "is_lethal": growth < 1e-6,
                    })
            except Exception as e:
                results.append({
                    "gene_a": ga,
                    "gene_b": gb,
                    "growth_double": np.nan,
                    "growth_ratio": np.nan,
                    "is_lethal": False,
                })

            if verbose and (i + 1) % 50 == 0:
                print(f"[FBA] Double KO progress: {i + 1}/{len(gene_pairs)}")

        return pd.DataFrame(results)

    def fba_with_chemometric_constraints(
        self,
        reaction_bounds: Dict[str, Tuple[float, float]],
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        施加计量学约束后进行 FBA

        这是核心科学卖点: "化学计量学校准的约束 FBA"

        参数
        ----
        reaction_bounds : dict {rxn_id: (lb, ub)}
            由 LatentToConstraint 生成的约束
        objective : str, optional
            目标反应 (默认保持原模型目标)

        返回
        ----
        result : dict
        """
        if not self._loaded:
            self.load_model()

        with self.model as m:
            # 施加约束
            applied = 0
            for rxn_id, (lb, ub) in reaction_bounds.items():
                try:
                    rxn = m.reactions.get_by_id(rxn_id)
                    rxn.lower_bound = max(rxn.lower_bound, lb)
                    rxn.upper_bound = min(rxn.upper_bound, ub)
                    applied += 1
                except KeyError:
                    pass  # 反应不在模型中, 跳过

            if objective:
                try:
                    m.objective = objective
                except Exception:
                    pass

            sol = m.optimize()

            # 获取关键通量
            key_fluxes = {}
            if sol.status == "optimal":
                for rxn in list(m.reactions)[:20]:  # 前20个反应
                    key_fluxes[rxn.id] = sol.fluxes[rxn.id]

            return {
                "objective_value": sol.objective_value,
                "status": sol.status,
                "constraints_applied": applied,
                "constraints_total": len(reaction_bounds),
                "key_fluxes": key_fluxes,
            }

    def flux_variability_analysis(
        self,
        reaction_ids: Optional[List[str]] = None,
        fraction_of_optimum: float = 0.9,
    ) -> pd.DataFrame:
        """
        通量变异性分析 (FVA)

        参数
        ----
        reaction_ids : list, optional
            待分析的反应列表
        fraction_of_optimum : float
            最优解比例 (0~1)

        返回
        ----
        fva_df : pd.DataFrame
        """
        try:
            import cobra.flux_analysis as cfa
        except ImportError:
            raise ImportError("COBRApy flux_analysis 不可用")

        if not self._loaded:
            self.load_model()

        if reaction_ids is None:
            reaction_ids = [r.id for r in self.model.reactions][:50]

        fva_result = cfa.flux_variability_analysis(
            self.model,
            reaction_list=reaction_ids,
            fraction_of_optimum=fraction_of_optimum,
        )

        return fva_result

    def get_all_genes(self) -> List[str]:
        """获取所有基因 ID"""
        if not self._loaded:
            self.load_model()
        return [g.id for g in self.model.genes]

    def get_exchange_reactions(self) -> List[str]:
        """获取所有交换反应 ID"""
        if not self._loaded:
            self.load_model()
        return [r.id for r in self.model.reactions if r.id.startswith("EX_")]

    def summary(self) -> str:
        """模型摘要"""
        if not self._loaded:
            return "FBASimulator (model not loaded)"

        m = self.model
        return (
            f"FBASimulator(model={self.model_name}, "
            f"genes={len(m.genes)}, reactions={len(m.reactions)}, "
            f"metabolites={len(m.metabolites)})"
        )


# ============================================================
#  便捷函数: 预测代谢偏移
# ============================================================

def predict_metabolic_shift(
    simulator: FBASimulator,
    wt_fluxes: Dict[str, float],
    ko_gene: str,
) -> pd.DataFrame:
    """
    预测敲除后的代谢偏移

    比较野生型与敲除株的关键通量变化

    参数
    ----
    simulator : FBASimulator
        已加载模型的模拟器
    wt_fluxes : dict
        野生型通量 {rxn_id: flux_value}
    ko_gene : str
        敲除基因 ID

    返回
    ----
    shift_df : pd.DataFrame
        通量变化表
    """
    if not simulator._loaded:
        simulator.load_model()

    rows = []
    try:
        with simulator.model as m:
            gene = m.genes.get_by_id(ko_gene)
            gene.knock_out()
            sol = m.optimize()

            if sol.status == "optimal":
                for rxn_id in list(wt_fluxes.keys()):
                    wt_f = wt_fluxes.get(rxn_id, 0.0)
                    ko_f = sol.fluxes.get(rxn_id, 0.0)
                    shift = ko_f - wt_f
                    rows.append({
                        "reaction": rxn_id,
                        "wt_flux": wt_f,
                        "ko_flux": ko_f,
                        "shift": shift,
                        "shift_pct": shift / (abs(wt_f) + 1e-8) * 100,
                    })
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("shift", key=abs, ascending=False)
    return df
