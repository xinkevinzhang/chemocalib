"""
ChemoCalib CLI —— 多块 PLS → 多约束 GEM → 主动选样 命令行工具
=====================================================================
阶段二核心 CLI: 一条命令跑通完整闭环

用法:
  chemocalib pipeline        # 跑完整 pipeline
  chemocalib mbpls           # 仅 MB-PLS 分析
  chemocalib constrain       # PLS 潜变量 → GEM 约束
  chemocalib knockout        # 虚拟双敲除
  chemocalib active-learn    # 主动学习选样
  chemocalib ode             # 动态 ODE 模拟
"""

import click
import sys
import os
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="0.2.0", prog_name="chemocalib")
def main():
    """
    ChemoCalib v0.2.0 — 多组学多块扩展 + 主动学习闭环

    化学计量学校准的约束代谢网络建模工具。
    """
    pass


@main.command()
@click.option("--n-samples", default=100, help="合成数据样本数")
@click.option("--n-components", default=5, help="潜变量数")
@click.option("--seed", default=42, help="随机种子")
@click.option("--output", default=None, help="输出 JSON 路径")
def mbpls(n_samples, n_components, seed, output):
    """
    多块 PLS 分析: 代谢组 + 转录组 + 蛋白组
    """
    from chemocalib.models.mbpls import MultiBlockPLS, generate_toy_multiblock_data

    console.print(
        Panel.fit("[bold cyan]多块 PLS 分析[/bold cyan]", border_style="cyan")
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("生成合成多组学数据...", total=None)
        blocks, y, feature_names = generate_toy_multiblock_data(
            n_samples=n_samples, seed=seed
        )
        progress.update(
            task, description=f"已生成数据: {blocks[0].shape[0]} 样本, 3 块"
        )

        task2 = progress.add_task("训练 MB-PLS 模型...", total=None)
        model = MultiBlockPLS(
            n_components=n_components,
            block_names=["代谢组", "转录组", "蛋白组"],
        )
        model.fit(blocks, y)
        progress.update(task2, description="MB-PLS 训练完成")

    # 打印结果
    rprint(model.summary())

    # VIP 驱动代谢物
    drivers = model.get_driving_metabolites(block_idx=0, top_k=10)
    table = Table(title="VIP Top-10 驱动代谢物 (代谢组)")
    table.add_column("索引", style="cyan")
    table.add_column("VIP 分数", style="green")
    for idx, vip in zip(drivers["indices"], drivers["vip_values"]):
        table.add_row(f"met_{idx}", f"{vip:.4f}")
    console.print(table)

    # 残差不确性
    residuals = model.residual_space(blocks)
    uncertainty = model.uncertainty_score(blocks)
    console.print(
        f"\n[bold]残差不确定性 (Top 5):[/bold] {np.sort(uncertainty)[::-1][:5].round(4)}"
    )

    if output:
        result = model.to_dict()
        result["uncertainty_top5"] = np.sort(uncertainty)[::-1][:5].tolist()
        with open(output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"[green]结果已保存到 {output}[/green]")


@main.command()
@click.option("--n-samples", default=100, help="样本数")
@click.option("--n-components", default=5, help="潜变量数")
@click.option("--model-name", default="textbook", help="GEM 模型名")
@click.option("--mode", default="soft", help="约束模式: soft/hard/adaptive")
@click.option("--output", default=None, help="输出路径")
def constrain(n_samples, n_components, model_name, mode, output):
    """
    PLS 潜变量 → GEM 约束映射
    """
    from chemocalib.models.mbpls import MultiBlockPLS, generate_toy_multiblock_data
    from chemocalib.gem.constraints import LatentToConstraint
    from chemocalib.gem.fba import FBASimulator

    console.print(
        Panel.fit("[bold cyan]潜变量 → GEM 约束映射[/bold cyan]", border_style="cyan")
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("生成数据 + 训练 MB-PLS...", total=None)
        blocks, y, feature_names = generate_toy_multiblock_data(n_samples=n_samples)
        model = MultiBlockPLS(
            n_components=n_components, block_names=["代谢组", "转录组", "蛋白组"]
        )
        model.fit(blocks, y)

        task2 = progress.add_task("加载 GEM 模型...", total=None)
        sim = FBASimulator(model_name=model_name)
        sim.load_model()

        task3 = progress.add_task("建立约束映射...", total=None)
        mapper = LatentToConstraint(scaling_mode=mode)
        gem_metabolites = sim.get_exchange_reactions()
        gem_met_ids = [m.replace("EX_", "") for m in gem_metabolites]
        mapper.build_feature_reaction_map(
            feature_names[0],
            gem_met_ids[: len(feature_names[0])],
            vip_scores=model.vip_scores[0],
        )

        task4 = progress.add_task("施加约束 + FBA...", total=None)
        latent_mean = model.super_scores.mean(axis=0)
        bounds = mapper.latent_to_bounds(latent_mean, n_components=3)
        fba_result = sim.fba_with_chemometric_constraints(bounds)

    # 输出
    table = Table(title="约束 → FBA 结果")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row(
        "约束施加数",
        f"{fba_result['constraints_applied']}/{fba_result['constraints_total']}",
    )
    table.add_row("目标值 (生物量)", f"{fba_result['objective_value']:.4f}")
    table.add_row("求解状态", fba_result["status"])
    console.print(table)

    console.print("\n[bold]施加的约束:[/bold]")
    df = mapper.to_dataframe()
    console.print(df.to_string(index=False))

    if output:
        result = {
            "fba": {k: v for k, v in fba_result.items() if k != "key_fluxes"},
            "constraints": df.to_dict(orient="records"),
        }
        with open(output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"[green]结果已保存到 {output}[/green]")


@main.command()
@click.option("--n-pairs", default=200, help="双敲除对数")
@click.option("--model-name", default="textbook", help="GEM 模型名")
@click.option("--strategy", default="exhaustive", help="设计策略: exhaustive/vip_top")
@click.option("--output", default=None, help="输出 CSV 路径")
def knockout(n_pairs, model_name, strategy, output):
    """
    虚拟双敲除实验: 生成 + 运行 in-silico 双敲除
    """
    from chemocalib.gem.fba import FBASimulator
    from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

    console.print(
        Panel.fit("[bold cyan]虚拟双敲除实验[/bold cyan]", border_style="cyan")
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("加载 GEM 模型...", total=None)
        sim = FBASimulator(model_name=model_name)
        sim.load_model()
        genes = sim.get_all_genes()

        task2 = progress.add_task(f"设计 {n_pairs} 组双敲除...", total=None)
        designer = DoubleKnockoutDesigner(
            gene_pool=genes,
            design_strategy=strategy,
        )
        designer.generate_pairs(n_pairs=n_pairs)

        task3 = progress.add_task("运行虚拟实验 (批量 FBA)...", total=None)
        results = designer.run_virtual_experiments(sim, verbose=False)

    # 分析
    analysis = designer.analyze_results()
    table = Table(title="虚拟双敲除实验结果")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("总实验数", str(analysis["n_total"]))
    table.add_row("致死 (生长率=0)", str(analysis["n_lethal"]))
    table.add_row(
        "非致死",
        str(analysis.get("n_nonlethal", analysis["n_total"] - analysis["n_lethal"])),
    )
    gs = analysis.get("growth_stats", {})
    table.add_row("平均生长率", f"{gs.get('mean', 0):.4f}")
    table.add_row("生长率标准差", f"{gs.get('std', 0):.4f}")
    console.print(table)

    # Top pairs
    top10 = results.nlargest(10, "growth_ratio")
    console.print("\n[bold]Top 10 生长率双敲除:[/bold]")
    console.print(top10[["gene_a", "gene_b", "growth_ratio"]].to_string(index=False))

    if output:
        results.to_csv(output, index=False)
        console.print(f"[green]结果已导出到 {output}[/green]")


@main.command()
@click.option("--n-samples", default=100, help="样本数")
@click.option("--n-select", default=10, help="选取数量")
@click.option(
    "--strategy", default="hybrid", help="采样策略: residual/entropy/diversity/hybrid"
)
@click.option("--output", default=None, help="输出 JSON 路径")
def active_learn(n_samples, n_select, strategy, output):
    """
    主动学习选样: 基于不确定性选下一个双敲除
    """
    from chemocalib.models.mbpls import MultiBlockPLS, generate_toy_multiblock_data
    from chemocalib.active_learning.uncertainty import UncertaintySampler
    from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

    console.print(Panel.fit("[bold cyan]主动学习选样[/bold cyan]", border_style="cyan"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("训练 MB-PLS...", total=None)
        blocks, y, feature_names = generate_toy_multiblock_data(n_samples=n_samples)
        model = MultiBlockPLS(
            n_components=5, block_names=["代谢组", "转录组", "蛋白组"]
        )
        model.fit(blocks, y)

        task2 = progress.add_task("计算残差空间...", total=None)
        residuals = model.residual_space(blocks)

        task3 = progress.add_task("主动学习采样...", total=None)
        sampler = UncertaintySampler(strategy=strategy)
        uncertainty = sampler.compute_uncertainty(residuals)

        # 模拟基因对
        n_genes = 50
        gene_pairs = [
            (f"G{i}", f"G{j}") for i in range(n_genes) for j in range(i + 1, n_genes)
        ]
        pair_subset = gene_pairs[: min(len(gene_pairs), 500)]

        # 选候选
        candidates = sampler.select_double_knockout_candidates(
            all_gene_pairs=pair_subset,
            pair_features=np.random.randn(len(pair_subset), 5),
            residuals=residuals,
            n_select=n_select,
            n_pool=200,
        )

    table = Table(title=f"主动学习 Top {n_select} 推荐 (策略: {strategy})")
    table.add_column("排名", style="cyan")
    table.add_column("基因A", style="yellow")
    table.add_column("基因B", style="yellow")
    table.add_column("不确定性", style="green")
    for _, row in candidates.iterrows():
        table.add_row(
            str(row["rank"]),
            row["gene_a"],
            row["gene_b"],
            f"{row['uncertainty']:.4f}",
        )
    console.print(table)

    if output:
        candidates.to_json(output, orient="records", force_ascii=False)
        console.print(f"[green]结果已保存到 {output}[/green]")


@main.command()
@click.option("--duration", default=50.0, help="模拟时长")
@click.option("--n-points", default=200, help="输出点数")
@click.option("--calibrate", is_flag=True, help="用潜变量校准 ODE 参数")
@click.option("--output", default=None, help="输出 CSV 路径")
def ode(duration, n_points, calibrate, output):
    """
    动态 ODE 模拟: 糖酵解节点动力学
    """
    from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
    from chemocalib.models.mbpls import generate_toy_multiblock_data

    console.print(
        Panel.fit("[bold cyan]糖酵解 ODE 动态模拟[/bold cyan]", border_style="cyan")
    )

    ode_model = GlycolysisODE(
        vmax_hk=1.0,
        vmax_pfk=1.2,
        vmax_pk=0.8,
        km_glc=1.0,
        km_g6p=0.5,
        km_fbp=0.3,
    )

    if calibrate:
        console.print("[dim]使用 PLS 潜变量校准 ODE 参数...[/dim]")
        blocks, y, _ = generate_toy_multiblock_data(n_samples=100)
        from chemocalib.models.mbpls import MultiBlockPLS

        mbpls = MultiBlockPLS(n_components=3)
        mbpls.fit(blocks, y)
        latent_mean = mbpls.super_scores.mean(axis=0)
        ode_model.calibrate_from_latent(latent_mean, n_component=0)

    result = ode_model.simulate(t_span=(0, duration), n_points=n_points)

    # 输出
    table = Table(title="ODE 模拟结果 (终点)")
    table.add_column("代谢物", style="cyan")
    table.add_column("稳态浓度", style="green")
    table.add_row("G6P", f"{result['G6P'][-1]:.4f}")
    table.add_row("FBP", f"{result['FBP'][-1]:.4f}")
    table.add_row("PYR", f"{result['PYR'][-1]:.4f}")
    console.print(table)

    # kcat 代理
    kcat = ode_model.extract_kcat_proxies()
    console.print(f"\n[bold]kcat 代理值:[/bold] {kcat}")

    if output:
        df = pd.DataFrame(
            {
                "t": result["t"],
                "G6P": result["G6P"],
                "FBP": result["FBP"],
                "PYR": result["PYR"],
                "v_HK": (
                    result["fluxes"]["v_HK"]
                    if isinstance(result["fluxes"]["v_HK"], np.ndarray)
                    else np.full_like(result["t"], result["fluxes"]["v_HK"])
                ),
                "v_PFK": result["fluxes"]["v_PFK"],
                "v_PK": result["fluxes"]["v_PK"],
            }
        )
        df.to_csv(output, index=False)
        console.print(f"[green]时间序列已保存到 {output}[/green]")


@main.command()
@click.option("--n-samples", default=100, help="合成数据样本数")
@click.option("--n-components", default=5, help="潜变量数")
@click.option("--n-pairs", default=150, help="双敲除对数 (轻薄本建议 100-200)")
@click.option("--n-select", default=10, help="主动学习选取数")
@click.option("--model-name", default="textbook", help="GEM 模型: textbook 或 iMM904")
@click.option("--mode", default="soft", help="约束模式: soft/hard/adaptive")
@click.option("--skip-ode", is_flag=True, help="跳过 ODE 模拟")
@click.option("--output-dir", default="./output", help="输出目录")
def pipeline(
    n_samples, n_components, n_pairs, n_select, model_name, mode, skip_ode, output_dir
):
    """
    完整闭环 Pipeline: MB-PLS → GEM约束 → 双敲除 → 主动选样 → ODE
    """
    import time
    from chemocalib.models.mbpls import MultiBlockPLS, generate_toy_multiblock_data
    from chemocalib.gem.constraints import LatentToConstraint
    from chemocalib.gem.fba import FBASimulator
    from chemocalib.active_learning.uncertainty import UncertaintySampler
    from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner
    from chemocalib.virtual_experiment.surrogate import SurrogateModel
    from chemocalib.dynamic_layer.ode_solver import GlycolysisODE

    os.makedirs(output_dir, exist_ok=True)

    console.print(
        Panel.fit(
            "[bold cyan]ChemoCalib 完整闭环 Pipeline[/bold cyan]\n"
            "多块 PLS → 多约束 GEM → 虚拟双敲除 → 主动选样 → ODE 动态层",
            border_style="cyan",
        )
    )

    start_time = time.time()
    results_summary = {}

    # =================================================================
    # Step 1: 多块 PLS 训练
    # =================================================================
    console.rule("[bold]Step 1/5: 多块 PLS 训练[/bold]")
    blocks, y, feature_names = generate_toy_multiblock_data(n_samples=n_samples)
    mbpls = MultiBlockPLS(
        n_components=n_components,
        block_names=["代谢组", "转录组", "蛋白组"],
    )
    mbpls.fit(blocks, y)
    rprint(mbpls.summary())
    results_summary["mbpls"] = mbpls.to_dict()

    # =================================================================
    # Step 2: 潜变量 → GEM 约束 + FBA
    # =================================================================
    console.rule("[bold]Step 2/5: 潜变量 → GEM 约束映射[/bold]")
    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    wt_result = sim.wild_type_fba()
    console.print(f"  野生型生物量: {wt_result['objective_value']:.4f}")

    gem_metabolites = sim.get_exchange_reactions()
    gem_met_ids = [m.replace("EX_", "") for m in gem_metabolites]
    vip_scores = mbpls.vip_scores[0] if mbpls.vip_scores else None

    mapper = LatentToConstraint(scaling_mode=mode)
    mapper.build_feature_reaction_map(
        feature_names[0],
        gem_met_ids[: len(feature_names[0])],
        vip_scores=vip_scores,
    )

    latent_mean = mbpls.super_scores.mean(axis=0)
    bounds = mapper.latent_to_bounds(latent_mean, n_components=3)
    fba_result = sim.fba_with_chemometric_constraints(bounds)
    console.print(
        f"  施加 {fba_result['constraints_applied']}/{fba_result['constraints_total']} 个约束"
    )
    console.print(f"  约束后生物量: {fba_result['objective_value']:.4f}")
    results_summary["constraint_fba"] = {
        "wt_biomass": wt_result["objective_value"],
        "constrained_biomass": fba_result["objective_value"],
        "constraints_applied": fba_result["constraints_applied"],
    }

    # =================================================================
    # Step 3: 虚拟双敲除
    # =================================================================
    console.rule("[bold]Step 3/5: 虚拟双敲除实验[/bold]")
    genes = sim.get_all_genes()
    designer = DoubleKnockoutDesigner(gene_pool=genes, design_strategy="exhaustive")
    designer.generate_pairs(n_pairs=n_pairs)
    dko_results = designer.run_virtual_experiments(sim, verbose=True)
    analysis = designer.analyze_results()

    table = Table(title="双敲除统计")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    for k, v in analysis.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                table.add_row(f"  {k}.{k2}", f"{v2:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)
    results_summary["knockout"] = analysis
    dko_results.to_csv(
        os.path.join(output_dir, "double_knockout_results.csv"), index=False
    )

    # =================================================================
    # Step 4: 主动学习选样
    # =================================================================
    console.rule("[bold]Step 4/5: 主动学习 —— 选最优双敲除候选[/bold]")
    residuals = mbpls.residual_space(blocks)
    sampler = UncertaintySampler(strategy="hybrid")

    # 构建候选基因对 (已有 FBA 结果的)
    gene_pairs = list(zip(dko_results["gene_a"], dko_results["gene_b"]))
    candidates = sampler.select_double_knockout_candidates(
        all_gene_pairs=gene_pairs,
        pair_features=dko_results[["growth_ratio"]].values,
        residuals=residuals,
        n_select=n_select,
        n_pool=min(200, len(gene_pairs)),
    )

    table = Table(title=f"主动学习推荐 Top {n_select} (给合作者真做)")
    table.add_column("排名", style="cyan")
    table.add_column("基因A", style="yellow")
    table.add_column("基因B", style="yellow")
    table.add_column("不确定性", style="green")
    for _, row in candidates.iterrows():
        table.add_row(
            str(row["rank"]), row["gene_a"], row["gene_b"], f"{row['uncertainty']:.4f}"
        )
    console.print(table)
    candidates.to_csv(
        os.path.join(output_dir, "active_learning_candidates.csv"), index=False
    )
    results_summary["active_learning"] = candidates.to_dict(orient="records")

    # =================================================================
    # Step 5: 动态 ODE (可选)
    # =================================================================
    if not skip_ode:
        console.rule("[bold]Step 5/5: 糖酵解 ODE 动态模拟[/bold]")
        ode_model = GlycolysisODE()
        ode_model.calibrate_from_latent(latent_mean, n_component=0)
        ode_result = ode_model.simulate(t_span=(0, 50), n_points=200)

        table = Table(title="ODE 稳态浓度")
        table.add_column("代谢物", style="cyan")
        table.add_column("终浓度", style="green")
        table.add_column("解析稳态", style="yellow")
        ss = ode_model.steady_state()
        for i, name in enumerate(["G6P", "FBP", "PYR"]):
            ss_val = f"{ss[i]:.4f}" if ss is not None else "N/A"
            table.add_row(name, f"{ode_result[name][-1]:.4f}", ss_val)
        console.print(table)

        ode_df = pd.DataFrame(
            {
                "t": ode_result["t"],
                "G6P": ode_result["G6P"],
                "FBP": ode_result["FBP"],
                "PYR": ode_result["PYR"],
            }
        )
        ode_df.to_csv(os.path.join(output_dir, "ode_timecourse.csv"), index=False)
        results_summary["ode"] = {
            "final_state": {n: float(ode_result[n][-1]) for n in ["G6P", "FBP", "PYR"]}
        }

    # =================================================================
    # 完成
    # =================================================================
    elapsed = time.time() - start_time
    console.rule(f"[bold green]Pipeline 完成! 耗时 {elapsed:.1f}s[/bold green]")

    # 保存摘要
    with open(os.path.join(output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\n[bold]输出目录: {output_dir}[/bold]")
    console.print(f"  - pipeline_summary.json")
    console.print(f"  - double_knockout_results.csv")
    console.print(f"  - active_learning_candidates.csv")
    if not skip_ode:
        console.print(f"  - ode_timecourse.csv")
    console.print(
        f"\n[bold cyan]研究身份: 我用化学计量学把多组学定量校准进代谢约束的那一层[/bold cyan]"
    )


if __name__ == "__main__":
    main()
