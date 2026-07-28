"""
动态 ODE 层 —— 糖酵解节点动力学
=====================================
用 ODE 描述关键代谢节点 (糖酵解),
以 PLS 拟合的速率常数初始化参数。

用途:
  - 补充 FBA 的稳态假设, 引入时间维度
  - 用 PLS 潜变量校准 ODE 参数 (kcat 等)
  - 模拟瞬态代谢响应

模型:
  简化糖酵解三节点:
    GLC --(HK)--> G6P --(PFK)--> FBP --(下游)--> PYR

  使用 Michaelis-Menten 动力学,
  方程用 scipy.integrate.solve_ivp 求解。
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple, Callable
import warnings


class GlycolysisODE:
    """
    糖酵解 ODE 模型

    三个核心代谢物: [G6P], [FBP], [PYR]
    GLC 视为恒定外部输入

    参数
    ----
    vmax_hk : float, 默认=1.0
        己糖激酶最大速率
    vmax_pfk : float, 默认=1.0
        磷酸果糖激酶最大速率
    vmax_pk : float, 默认=1.0
        丙酮酸激酶最大速率
    km_glc : float, 默认=1.0
        HK 的 Km (葡萄糖)
    km_g6p : float, 默认=1.0
        PFK 的 Km (G6P)
    km_fbp : float, 默认=1.0
        PK 的 Km (FBP)
    glc_input : float, 默认=10.0
        恒定葡萄糖输入浓度
    """

    def __init__(
        self,
        vmax_hk: float = 1.0,
        vmax_pfk: float = 1.0,
        vmax_pk: float = 1.0,
        km_glc: float = 1.0,
        km_g6p: float = 1.0,
        km_fbp: float = 1.0,
        glc_input: float = 10.0,
    ):
        # 动力学参数
        self.vmax_hk = vmax_hk
        self.vmax_pfk = vmax_pfk
        self.vmax_pk = vmax_pk
        self.km_glc = km_glc
        self.km_g6p = km_g6p
        self.km_fbp = km_fbp
        self.glc_input = glc_input

        # 模拟结果
        self.t: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._success = False

    def _ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE 右端项: d[metabolites]/dt

        y[0] = [G6P], y[1] = [FBP], y[2] = [PYR]
        """
        G6P, FBP, PYR = y[0], y[1], y[2]

        # Michaelis-Menten 通量
        v_hk = self.vmax_hk * self.glc_input / (self.km_glc + self.glc_input)
        v_pfk = self.vmax_pfk * G6P / (self.km_g6p + G6P)
        v_pk = self.vmax_pk * FBP / (self.km_fbp + FBP)

        # 物料平衡
        dG6P = v_hk - v_pfk
        dFBP = v_pfk - v_pk
        dPYR = v_pk

        return np.array([dG6P, dFBP, dPYR])

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 50.0),
        y0: Optional[List[float]] = None,
        n_points: int = 200,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        模拟糖酵解动力学

        参数
        ----
        t_span : tuple
            时间范围 (t_start, t_end)
        y0 : list of float, optional
            初始浓度 [G6P, FBP, PYR], 默认 [0, 0, 0]
        n_points : int
            输出时间点数

        返回
        ----
        result : dict
            {"t": array, "G6P": array, "FBP": array, "PYR": array, "fluxes": {...}}
        """
        if y0 is None:
            y0 = [0.0, 0.0, 0.0]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            self._ode_rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            **kwargs,
        )

        self.t = sol.t
        self.y = sol.y
        self._success = sol.success

        # 计算各时间点的通量
        fluxes = self.compute_fluxes_at_time(self.y)

        return {
            "t": sol.t,
            "G6P": sol.y[0],
            "FBP": sol.y[1],
            "PYR": sol.y[2],
            "fluxes": fluxes,
            "success": sol.success,
        }

    def compute_fluxes_at_time(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """计算给定浓度时的各反应通量"""
        G6P, FBP = y[0], y[1]

        v_hk = self.vmax_hk * self.glc_input / (self.km_glc + self.glc_input)
        v_pfk = self.vmax_pfk * G6P / (self.km_g6p + G6P)
        v_pk = self.vmax_pk * FBP / (self.km_fbp + FBP)

        return {
            "v_HK": np.full_like(G6P, v_hk) if np.isscalar(v_hk) else v_hk,
            "v_PFK": v_pfk,
            "v_PK": v_pk,
        }

    def steady_state(self) -> Optional[np.ndarray]:
        """
        计算稳态浓度 (解析解)

        对简化模型:
          v_hk = v_pfk = v_pk (稳态通量相等)
          G6P_ss = km_g6p * v_hk / (vmax_pfk - v_hk)
          FBP_ss = km_fbp * v_hk / (vmax_pk - v_hk)
        """
        v_hk = self.vmax_hk * self.glc_input / (self.km_glc + self.glc_input)

        if v_hk >= self.vmax_pfk or v_hk >= self.vmax_pk:
            return None  # 无法达到稳态

        G6P_ss = self.km_g6p * v_hk / (self.vmax_pfk - v_hk)
        FBP_ss = self.km_fbp * v_hk / (self.vmax_pk - v_hk)
        # PYR 稳态取决于下游消耗, 设为零
        PYR_ss = 0.0

        return np.array([G6P_ss, FBP_ss, PYR_ss])

    def calibrate_from_latent(
        self,
        latent_scores: np.ndarray,
        n_component: int = 0,
        scale_vmax: float = 2.0,
        scale_km: float = 1.0,
    ):
        """
        用 PLS 潜变量校准 ODE 参数

        映射逻辑:
          潜变量分量 → 酶活性调节因子 → Vmax/Km 缩放

        参数
        ----
        latent_scores : np.ndarray (n_components,)
            单个样本的潜变量得分
        n_component : int
            使用的分量索引
        scale_vmax : float
            Vmax 缩放幅度
        scale_km : float
            Km 缩放幅度
        """
        if latent_scores.ndim > 1:
            lv = latent_scores.flatten()
        else:
            lv = latent_scores

        if n_component < len(lv):
            factor = lv[n_component]
        else:
            factor = 0.0

        # sigmoid 映射到 [0.5, 2.0] 范围
        modulation = 1.0 + scale_vmax * np.tanh(factor)

        self.vmax_hk *= modulation
        self.vmax_pfk *= modulation
        self.vmax_pk *= modulation

        # Km 也微调
        km_mod = 1.0 + scale_km * 0.1 * factor
        self.km_glc *= km_mod
        self.km_g6p *= km_mod
        self.km_fbp *= km_mod

        print(
            f"[ODE] 已校准: Vmax × {modulation:.2f}, Km × {km_mod:.2f}"
        )

    def extract_kcat_proxies(self) -> Dict[str, float]:
        """
        提取 kcat 代理值 (用于反向标定)

        返回
        ----
        kcat_proxies : dict
            {enzyme: estimated_kcat_ratio}
        """
        v_hk = self.vmax_hk * self.glc_input / (self.km_glc + self.glc_input)
        return {
            "HK_kcat_proxy": float(v_hk),
            "PFK_kcat_proxy": float(self.vmax_pfk / self.km_g6p),
            "PK_kcat_proxy": float(self.vmax_pk / self.km_fbp),
        }

    def summary(self) -> str:
        """模型摘要"""
        params = (
            f"Vmax=(HK:{self.vmax_hk:.2f}, PFK:{self.vmax_pfk:.2f}, PK:{self.vmax_pk:.2f}), "
            f"Km=(GLC:{self.km_glc:.2f}, G6P:{self.km_g6p:.2f}, FBP:{self.km_fbp:.2f})"
        )
        return f"GlycolysisODE({params})"
