from pathlib import Path

import numpy as np
import pandas as pd
import patchworklib as pw
import plotnine as p9
import torch
from einops import rearrange
from hydra import compose, initialize
from mizani.formatters import scientific_format
from torch import Tensor

from deep_pdes.datasets.pme import Stefan


def main():
    heat_plots()
    pme_plots()
    stefan_plots()
    linear_advection_plots()
    burgers_plots()


def heat_plots():
    cfg = get_cfg_for_experiment("3b_heat_var_c")
    plot_dir = Path(cfg.base_dir).parent / "paper_plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    true_cons_df["model"] = "True"
    true_cons_df["trap"] = true_cons_df["true"]
    cons_df = pd.concat([cons_df, true_cons_df])
    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "True"]
    cons_df_indomain = (
        cons_df.loc[np.isclose(cons_df.param, 5)]
        .loc[lambda d: np.isin(d.model, models)]
        .assign(model=lambda d: pd.Categorical(d.model.astype(str), models))
        .groupby(["t_idx", "model"])
        .agg(
            trap_sd=pd.NamedAgg("trap", np.std),
            trap=pd.NamedAgg("trap", np.mean),
            true=pd.NamedAgg("true", np.mean),
        )
        .reset_index()
        .assign(t=lambda df: df.t_idx / 200)
        .assign(trap_se=lambda d: d.trap_sd / np.sqrt(50))
        .assign(error=lambda d: d.trap - d.true)
    )
    colors = cfg.analysis.colors
    cons_df_indomain.loc[lambda d: np.isclose(d.t, 0.5)].to_csv(plot_dir / "heat_cons.csv")

    cons_plot = (
        p9.ggplot(cons_df_indomain, p9.aes(x="t"))
        + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.geom_point(
            p9.aes(y="trap", color="model", shape="model"),
            cons_df_indomain.loc[lambda d: d.t_idx % 20 == 0],
            size=5,
        )
        + p9.scale_color_manual(colors)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.ylab("Mass")
        + p9.labs(color="Model", shape="Model")
    )
    for ext in ("pdf", "png"):
        cons_plot.save(
            plot_dir / f"heat_cons_plot.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.cons_plot_width,
            height=cfg.analysis.cons_plot_height,
        )

    plot_df = _get_plot_df(cfg)
    true_df = _get_true_df(cfg)
    fid = cfg.analysis.fids_of_interest[0]
    t_of_interest = cfg.analysis.t_of_interest
    assert len(t_of_interest) == 1, "only look at one time point"
    for df in (plot_df, true_df):
        df["t_of_interest"] = False
        df["t_label"] = ""
        for t in t_of_interest:
            df.loc[np.isclose(df.t, t), "t_of_interest"] = True
            df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
    plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
    true_df_at_ts = true_df.loc[true_df.t_of_interest]

    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP"]
    plot_df_f_id = (
        plot_df_at_ts.loc[plot_df_at_ts.f_id == fid]
        .assign(param=lambda df: df["param"].astype(str))
        .loc[lambda df: np.isin(df["model"], models)]
        .assign(model=lambda df: pd.Categorical(df["model"].astype(str), models, ordered=True))
    )
    true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == fid].assign(
        param=lambda df: df["param"].astype(str)
    )
    true_df_f_id["model_type"] = "True solution"

    colors = cfg.analysis.colors
    plot_at_ts = (
        p9.ggplot(plot_df_f_id, mapping=p9.aes(x="x"))
        + p9.geom_line(p9.aes(y="u", color="model", group="model"), size=1)
        + p9.geom_line(
            p9.aes(y="u", linetype="model_type"), data=true_df_f_id, color="black", size=1.5
        )
        + p9.scale_linetype_manual(["dashed"])
        + p9.facet_wrap("~param")
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
        )
    )
    plot_at_ts.save(
        plot_dir / f"heat_time_plot.pdf",
        width=12,
        height=4,
        dpi=cfg.analysis.dpi,
    )

    error_df_f_id = (
        plot_df_f_id.set_index(["param", "f_id", "x", "t"])
        .join(
            true_df_f_id.set_index(["param", "f_id", "x", "t"]),
            on=["param", "f_id", "x", "t"],
            rsuffix="_true",
        )
        .reset_index()
        .assign(x_id=lambda df: (df.x / np.pi * 100).astype(int))
    )

    def error_plot_at_ts_labeller(x):
        if x in ("1", "5"):
            return f"k = {x}"
        else:
            return x

    solution_prof_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
        + p9.geom_point(
            p9.aes(color="model", group="model", shape="model"),
            error_df_f_id.loc[lambda df: df.x_id % 20 == 0],
            size=2,
        )
        + p9.geom_line(p9.aes(y="u_true", linetype="model_type"), color="black", size=1)
        + p9.geom_ribbon(
            p9.aes(
                ymin="-3 * u_sd + u",
                ymax="3 * u_sd + u",
                color="model",
                fill="model",
                group="model",
            ),
            alpha=0.1,
        )
        + p9.facet_grid("param~model", labeller=error_plot_at_ts_labeller)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["dashed"])
        + p9.labs(linetype="")
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, 0),
            legend_title=p9.element_blank(),
        )
        + p9.guides(fill=None, color=None, shape=None)
    )
    solution_prof_at_ts.save(
        plot_dir / f"heat_solution_profile.pdf",
        width=12,
        height=6,
        dpi=500,
    )

    error_plot_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u - u_true"))
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_grid("model~param")
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
    )
    error_plot_at_ts.save(
        plot_dir / f"heat_error_time_plot.pdf",
        width=9,
        height=12,
        dpi=500,
    )


def pme_plots():
    cfg = get_cfg_for_experiment("1b_pme_var_m")
    plot_dir = Path(cfg.base_dir).parent / "paper_plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    _pme_limiting_plots(cfg, plot_dir)

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    true_cons_df["model"] = "True"
    true_cons_df["trap"] = true_cons_df["true"]
    cons_df = pd.concat([cons_df, true_cons_df])
    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "True"]
    for param in (1, 3, 6):
        cons_df_indomain = (
            cons_df.loc[np.isclose(cons_df.param, param)]
            .loc[lambda d: np.isin(d.model, models)]
            # .assign(t_idx = lambda d: (d.t * 200).astype(int))
            .assign(model=lambda d: pd.Categorical(d.model.astype(str), models))
            .groupby(["t_idx", "model"])
            .agg(
                trap_sd=pd.NamedAgg("trap", np.std),
                trap=pd.NamedAgg("trap", np.mean),
                true=pd.NamedAgg("true", np.mean),
            )
            .reset_index()
            .assign(t=lambda df: df.t_idx / 200)
            .assign(trap_se=lambda d: d.trap_sd / np.sqrt(50))
            .assign(error=lambda d: d.trap - d.true)
        )
        # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 1)]
        colors = cfg.analysis.colors
        cons_df_indomain.loc[lambda d: np.isclose(d.t, 0.5)].to_csv(
            plot_dir / f"pme_cons_{param}.csv"
        )

    plot_df = _get_plot_df(cfg)
    true_df = _get_true_df(cfg)
    fid = cfg.analysis.fids_of_interest[0]
    t_of_interest = cfg.analysis.t_of_interest
    assert len(t_of_interest) == 1, "only look at one time point"
    for df in (plot_df, true_df):
        df["t_of_interest"] = False
        df["t_label"] = ""
        for t in t_of_interest:
            df.loc[np.isclose(df.t, t), "t_of_interest"] = True
            df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
    plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
    true_df_at_ts = true_df.loc[true_df.t_of_interest]

    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "ProbConserv-ANP (w/diff)"]

    plot_df_f_id = (
        plot_df_at_ts.loc[plot_df_at_ts.f_id == fid]
        .assign(param=lambda df: df["param"].astype(str))
        .loc[lambda df: np.isin(df["model"], models)]
        .assign(model=lambda df: pd.Categorical(df["model"].astype(str), models, ordered=True))
    )
    true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == fid].assign(
        param=lambda df: df["param"].astype(str)
    )

    # plot_df_f_id_outofdomain = plot_df_f_id.loc[np.isclose(plot_df_f_id.param, 0.5)]
    # true_df_f_id_outofdomain = true_df_f_id.loc[np.isclose(true_df_f_id.param, 0.5)]
    colors = cfg.analysis.colors
    plot_at_ts = (
        p9.ggplot(plot_df_f_id, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1)
        + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~param")
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        # + p9.guides(color=None, fill=None)
    )
    plot_at_ts.save(
        plot_dir / f"pme_time_plot.pdf",
        width=12,
        height=4,
        dpi=cfg.analysis.dpi,
    )

    error_df_f_id = (
        plot_df_f_id.set_index(["param", "f_id", "x", "t"])
        .join(
            true_df_f_id.set_index(["param", "f_id", "x", "t"]),
            on=["param", "f_id", "x", "t"],
            rsuffix="_true",
        )
        .reset_index()
        .assign(x_id=lambda df: (df.x * 200).astype(int))
    )

    def error_plot_at_ts_labeller(x):
        if x in ("1", "3", "6"):
            return f"m = {x}"
        else:
            return x

    error_df_f_id["model_type"] = "True solution"
    solution_prof_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
        + p9.geom_point(
            p9.aes(color="model", group="model", shape="model"),
            error_df_f_id.loc[lambda df: df.x_id % 20 == 0],
            size=2,
        )
        + p9.geom_line(p9.aes(y="u_true", linetype="model_type"), color="black", size=1)
        + p9.geom_ribbon(
            p9.aes(
                ymin="-3 * u_sd + u",
                ymax="3 * u_sd + u",
                color="model",
                fill="model",
                group="model",
            ),
            alpha=0.1,
        )
        + p9.facet_grid("param~model", labeller=error_plot_at_ts_labeller)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["dashed"])
        + p9.labs(linetype="")
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, 0),
            legend_title=p9.element_blank(),
        )
        + p9.guides(fill=None, color=None, shape=None)
    )
    solution_prof_at_ts.save(
        plot_dir / f"pme_solution_profile.pdf",
        width=14,
        height=7,
        dpi=500,
    )

    error_plot_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u - u_true"))
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_grid("model~param")
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
        )
        + p9.guides(color=None, fill=None)
    )
    error_plot_at_ts.save(
        plot_dir / f"pme_error_time_plot.pdf",
        width=9,
        height=12,
        dpi=500,
    )

    # error_df_f_id.loc[lambda df: df.param == 1]
    error_plot_at_ts_m1 = (
        p9.ggplot(
            error_df_f_id.loc[lambda df: df.param == "1"], mapping=p9.aes(x="x", y="u - u_true")
        )
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~model", ncol=2)
        + p9.geom_hline(yintercept=0, linetype="dashed")
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True), axis_title=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.ylab("$\\hat u - u$")
    )
    error_plot_at_ts_m1.save(
        plot_dir / f"pme_error_time_plot_m1.pdf",
        width=5,
        height=7.5,
        dpi=500,
    )

    error_plot_at_ts_m3 = (
        p9.ggplot(
            error_df_f_id.loc[lambda df: df.param == "3"], mapping=p9.aes(x="x", y="u - u_true")
        )
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~model", ncol=2)
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.geom_hline(yintercept=0, linetype="dashed")
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True), axis_title=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.ylab("$\\hat u - u$")
    )
    error_plot_at_ts_m3.save(
        plot_dir / f"pme_error_time_plot_m3.pdf",
        width=5,
        height=7.5,
        dpi=500,
    )

    error_plot_at_ts_m6 = (
        p9.ggplot(
            error_df_f_id.loc[lambda df: df.param == "6"], mapping=p9.aes(x="x", y="u - u_true")
        )
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~model", ncol=2)
        # + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.geom_hline(yintercept=0, linetype="dashed")
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True), axis_title=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.ylab("$\\hat u - u$")
    )
    error_plot_at_ts_m6.save(
        plot_dir / f"pme_error_time_plot_m6.pdf",
        width=5,
        height=7.5,
        dpi=500,
    )

    error_plot_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u - u_true"))
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.2,
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_grid("param~model", labeller=error_plot_at_ts_labeller)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.geom_hline(yintercept=0, linetype="dashed")
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True), axis_title=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.ylab("$\\hat u - u$")
    )
    error_plot_at_ts.save(
        plot_dir / f"pme_error_time_plot_horiz.pdf",
        width=14,
        height=7,
        dpi=500,
    )

    print("here")


def _pme_limiting_plots(cfg, plot_dir):
    mse_at_t_df = _get_mse_at_t_df(cfg)
    limiting_precisions = {
        "PhysNP(1e-6)": 1e-6,
        "PhysNP(1e-3)": 1e-3,
        "PhysNP(1e-2)": 1e-2,
        "PhysNP(1e-1)": 1e-1,
        "PhysNP(0)": 1.0,
        "PhysNP(1e1)": 1e1,
        "PhysNP(1e2)": 1e2,
        "PhysNP(1e3)": 1e3,
        "PhysNP(1e4)": 1e4,
        "PhysNP(1e5)": 1e5,
        "PhysNP(1e6)": 1e6,
        "PhysNP(1e7)": 1e7,
        "PhysNP(1e8)": 1e8,
    }
    x = (
        mse_at_t_df.loc[lambda r: np.isin(r["model"], list(limiting_precisions.keys()))].loc[
            lambda r: np.isclose(r["t"], 0.5)
        ]
        # .loc[lambda r: r["param"] == 3]
    )
    x["precis"] = x["model"].map(limiting_precisions)

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    x_cons = (
        cons_df.loc[lambda r: np.isin(r["model"], list(limiting_precisions.keys()))].loc[
            lambda r: np.isclose(r["t"], 0.5)
        ]
        # .loc[lambda r: r["param"] == 3]
    )
    x_cons["precis"] = x_cons["model"].map(limiting_precisions)
    x_cons = (
        x_cons.assign(sqerr=lambda df: df.error.pow(2))
        .groupby(["param", "t_idx", "precis"])
        .agg(cons_mse=pd.NamedAgg("sqerr", np.mean))
        .reset_index()
    )

    def error_plot_at_ts_labeller(x):
        if x in ("1", "3", "6"):
            return f"m = {x}"
        else:
            return x

    cons_limiting = (
        p9.ggplot(x_cons, p9.aes(x="precis", y="cons_mse"))
        + p9.geom_line()
        + p9.geom_point()
        # + p9.facet_wrap("~param", scales="free_y")
        + p9.facet_wrap("~param", labeller=error_plot_at_ts_labeller)
        + p9.scale_x_log10()
        + p9.theme_bw()
        + p9.xlab("Precision $1/\\sigma_G^2$")
        + p9.ylab("CE$^2$")
        + p9.theme(subplots_adjust={"wspace": 0.30})
        + p9.scale_y_continuous(labels=scientific_format(digits=2))
    )
    cons_limiting.save(plot_dir / "pme_limiting_cons.pdf", width=10, height=3, dpi=cfg.analysis.dpi)

    mse_limiting = (
        p9.ggplot(x, p9.aes(x="precis", y="MSE"))
        + p9.geom_line()
        + p9.geom_point()
        # + p9.facet_wrap("~param", scales="free_y", labeller=error_plot_at_ts_labeller)
        + p9.facet_wrap("~param", labeller=error_plot_at_ts_labeller, scales="free_y")
        # + p9.facet_wrap("~param", scales="free_y", labeller=error_plot_at_ts_labeller)
        + p9.scale_x_log10()
        + p9.theme_bw()
        + p9.xlab("Precision $1/\\sigma_G^2$")
        + p9.ylab("MSE")
        + p9.theme(subplots_adjust={"wspace": 0.30})
        + p9.scale_y_continuous(labels=scientific_format(digits=2))
    )
    mse_limiting.save(plot_dir / "pme_limiting_mse.pdf", width=10, height=3, dpi=cfg.analysis.dpi)

    ll_limiting = (
        p9.ggplot(x, p9.aes(x="precis", y="loglik"))
        + p9.geom_line()
        + p9.geom_point()
        # + p9.facet_wrap("~param")
        # + p9.facet_wrap("~param", scales="free_y")
        + p9.facet_wrap("~param", labeller=error_plot_at_ts_labeller)
        + p9.scale_x_log10()
        + p9.theme_bw()
        + p9.xlab("Precision $1/\\sigma_G^2$")
        + p9.ylab("Log-likelihood")
        + p9.theme(subplots_adjust={"wspace": 0.30})
    )
    ll_limiting.save(plot_dir / "pme_limiting_ll.pdf", width=10, height=3, dpi=cfg.analysis.dpi)
    print("hello")


def stefan_plots():
    cfg = get_cfg_for_experiment("2b_stefan_var_p")
    plot_dir = Path(cfg.base_dir).parent / "paper_plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    nice_names = cfg.analysis.nice_names
    colors = cfg.analysis.colors
    t = cfg.analysis.t_of_interest[0]
    fid = cfg.analysis.fids_of_interest[0]

    shock_df = _make_shock_df(cfg)
    min_shock, max_shock = _get_shock_range(cfg)
    shock_df_in_domain = shock_df.loc[np.isclose(shock_df.pstar, 0.6)]
    true_shock_value_in_domain = _get_true_shock_value(cfg, t, 0.6)
    true_shock_df = pd.DataFrame(
        {"true_shock": [true_shock_value_in_domain], "model_type": ["True shock"]}
    )
    shock_plot_in_domain = (
        p9.ggplot(shock_df_in_domain, p9.aes(x="shock_position"))  # noqa: WPS221
        + p9.geom_histogram(p9.aes(color="model", fill="model"), bins=15)
        + p9.geom_vline(
            p9.aes(xintercept="true_shock", linetype="model_type"), true_shock_df, size=1
        )
        # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=50, alpha=0.2)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["solid"])
        # + p9.scale_x_continuous(limits=(0.1, 0.4))
        + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
        + p9.facet_wrap("~model", ncol=4)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, -0.1),
            # axis_text_x=p9.element_text(size=9)
            # legend_position="top",
            # legend_title=p9.element_blank(),
            # axis_title_y=p9.element_blank(),
        )
        + p9.xlab("Shock position")
        + p9.ylab("Count")
        + p9.labs(linetype="")
        + p9.guides(color=None, fill=None)
    )
    for ext in ("pdf", "png"):
        shock_plot_in_domain.save(
            plot_dir / f"shock_plot_in_domain_fid={fid}_t={t:.3f}.{ext}",
            # width=cfg.analysis.shock_plot_width,
            # height=cfg.analysis.shock_plot_height,
            width=cfg.analysis.time_plot_width,
            height=cfg.analysis.time_plot_height,
            dpi=cfg.analysis.dpi,
        )
        shock_plot_in_domain.save(
            plot_dir / f"shock_plot_in_domain_fid={fid}_t={t:.3f}_BIG.{ext}",
            # width=cfg.analysis.shock_plot_width,
            # height=cfg.analysis.shock_plot_height,
            width=cfg.analysis.time_plot_width * 2,
            height=cfg.analysis.time_plot_height * 2,
            dpi=cfg.analysis.dpi,
        )

    # shock_df_out_of_domain = shock_df.loc[np.isclose(shock_df.pstar, 0.5)]
    # true_shock_value_out_of_domain = _get_true_shock_value(cfg, t, 0.5)
    # shock_plot_out_of_domain = (
    #     p9.ggplot(
    #         shock_df_out_of_domain, p9.aes(x="shock_position", color="model", fill="model")
    #     )  # noqa: WPS221
    #     + p9.geom_histogram()
    #     + p9.geom_vline(xintercept=true_shock_value_out_of_domain)
    #     # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=50, alpha=0.2)
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     + p9.facet_wrap("~model")
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(
    #         strip_text=p9.element_text(usetex=True),
    #         legend_position="top",
    #         legend_title=p9.element_blank(),
    #         axis_title_y=p9.element_blank(),
    #     )
    #     + p9.xlab("Shock position")
    #     + p9.guides(color=None, fill=None)
    # )
    # for ext in ("pdf", "png"):
    #     shock_plot_out_of_domain.save(
    #         plot_dir / f"shock_plot_out_of_domain_fid={fid}_t={t:.3f}.{ext}",
    #         width=cfg.analysis.time_plot_width,
    #         height=cfg.analysis.time_plot_height,
    #         # width=cfg.analysis.shock_plot_width,
    #         # height=cfg.analysis.shock_plot_height,
    #         dpi=cfg.analysis.dpi,
    #     )

    # Conservation / MSE (in-domain)
    plot_df = _get_plot_df(cfg)
    true_df = _get_true_df(cfg)
    fid = cfg.analysis.fids_of_interest[0]
    t_of_interest = cfg.analysis.t_of_interest
    assert len(t_of_interest) == 1, "only look at one time point"
    for df in (plot_df, true_df):
        df["t_of_interest"] = False
        df["t_label"] = ""
        for t in t_of_interest:
            df.loc[np.isclose(df.t, t), "t_of_interest"] = True
            df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
    plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
    true_df_at_ts = true_df.loc[true_df.t_of_interest]

    plot_df_f_id = plot_df_at_ts.loc[plot_df_at_ts.f_id == fid]
    true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == fid]

    # plot_df_f_id_outofdomain = plot_df_f_id.loc[np.isclose(plot_df_f_id.param, 0.5)]
    # true_df_f_id_outofdomain = true_df_f_id.loc[np.isclose(true_df_f_id.param, 0.5)]
    # plot_at_ts_outofdomain = (
    #     p9.ggplot(plot_df_f_id_outofdomain, mapping=p9.aes(x="x", y="u"))
    #     + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
    #     + p9.geom_line(data=true_df_f_id_outofdomain, color="black", linetype="dashed", size=1)
    #     # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=1, alpha=0.2)
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     + p9.scale_x_continuous(limits=(0, 0.5))
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(strip_text=p9.element_text(usetex=True))
    #     + p9.guides(color=None, fill=None)
    # )
    # plot_at_ts_outofdomain_facet = (
    #     plot_at_ts_outofdomain
    #     + p9.facet_wrap("~model")
    #     + p9.geom_ribbon(
    #         p9.aes(
    #             ymin="u - 3 * u_sd", ymax="u + 3 * u_sd", color="model", fill="model", group="model"
    #         ),
    #         alpha=0.1,
    #     )
    # )
    # plot_at_ts_outofdomain_facet.save(
    #     plot_dir / f"stefan_time_plot_outofdomain_fid={fid}_t={t:.2f}.pdf",
    #     width=cfg.analysis.time_plot_width,
    #     height=cfg.analysis.time_plot_height,
    #     dpi=cfg.analysis.dpi,
    # )
    # plot_at_ts_outofdomain_all = plot_at_ts_outofdomain
    # plot_at_ts_outofdomain_all.save(
    #     plot_dir / f"stefan_time_plot_outofdomain_all_fid={fid}_t={t:.2f}.pdf",
    #     width=cfg.analysis.time_plot_width / 1.5,
    #     height=cfg.analysis.time_plot_height / 1.5,
    #     dpi=cfg.analysis.dpi,
    # )

    plot_df_f_id_indomain = plot_df_f_id.loc[np.isclose(plot_df_f_id.param, 0.6)].assign(
        x_idx=lambda d: (d.x * 200).astype(int)
    )
    true_df_f_id_indomain = true_df_f_id.loc[np.isclose(true_df_f_id.param, 0.6)].assign(
        x_idx=lambda d: (d.x * 200).astype(int)
    )
    true_df_f_id_indomain["model_type"] = "True solution"
    plot_at_ts_indomain = (
        p9.ggplot(plot_df_f_id_indomain, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
        + p9.geom_point(
            p9.aes(color="model", group="model", shape="model"),
            plot_df_f_id_indomain.loc[lambda d: d.x_idx % 5 == 0],
            size=3,
        )
        + p9.geom_line(
            p9.aes(linetype="model_type"), data=true_df_f_id_indomain, color="black", size=1
        )
        # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=1, alpha=0.2)
        + p9.scale_x_continuous(limits=(0.1, 0.35))
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["dashed"])
        + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, -0.1),
            # axis_text_x=p9.element_text(size=9),
        )
        + p9.guides(color=None, fill=None, shape=None)
        + p9.labs(linetype="")
    )
    plot_at_ts_indomain_facet = (
        plot_at_ts_indomain
        + p9.facet_wrap("~model", ncol=4)
        + p9.geom_ribbon(
            p9.aes(
                ymin="u - 3 * u_sd", ymax="u + 3 * u_sd", color="model", fill="model", group="model"
            ),
            alpha=0.1,
        )
    )
    plot_at_ts_indomain_facet.save(
        plot_dir / f"stefan_time_plot_indomain_fid={fid}_t={t:.2f}.pdf",
        width=cfg.analysis.time_plot_width,
        height=cfg.analysis.time_plot_height,
        dpi=cfg.analysis.dpi,
    )
    plot_at_ts_indomain_facet.save(
        plot_dir / f"stefan_time_plot_indomain_fid={fid}_t={t:.2f}_BIG.pdf",
        width=cfg.analysis.time_plot_width * 2,
        height=cfg.analysis.time_plot_height * 2,
        dpi=cfg.analysis.dpi,
    )
    # plot_at_ts_indomain_all = plot_at_ts_indomain
    # plot_at_ts_indomain_all.save(
    #     plot_dir / f"stefan_time_plot_indomain_all_fid={fid}_t={t:.2f}.pdf",
    #     width=cfg.analysis.time_plot_width / 2,
    #     height=cfg.analysis.time_plot_height / 2,
    #     dpi=cfg.analysis.dpi,
    # )

    error_df_f_id = (
        plot_df_f_id_indomain.set_index(["param", "f_id", "x", "t"])
        .join(
            true_df_f_id_indomain.set_index(["param", "f_id", "x", "t"]),
            on=["param", "f_id", "x", "t"],
            rsuffix="_true",
        )
        .reset_index()
    )
    error_plot_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u - u_true"))
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.1,
            linetype="dashed",
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~model")
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.xlab("x")
        + p9.ylab("Error")
    )
    error_plot_at_ts.save(
        plot_dir / f"stefan_error_time_plot.pdf",
        width=cfg.analysis.time_plot_width,
        height=cfg.analysis.time_plot_height,
        dpi=cfg.analysis.dpi,
    )

    mse_at_t_df = _get_mse_at_t_df(cfg)
    mse_at_t_df_indomain = mse_at_t_df.loc[np.isclose(mse_at_t_df.param, 0.6)]
    mse_at_t_plot = (
        p9.ggplot(mse_at_t_df_indomain, p9.aes(x="t", y="MSE", color="model"))  # noqa: WPS221
        + p9.geom_smooth(span=0.05, se=False)
        # + p9.facet_grid("~param")
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        # + p9.scale_color_hue(labels=nice_names.values())
        # + p9.guides(color=None)
        # + p9.theme(
        #     strip_text=p9.element_text(usetex=True), legend_position="bottom", legend_title=None
        # )
    )
    for ext in ("pdf", "png"):
        mse_at_t_plot.save(
            plot_dir / f"stefan_mse_over_time.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.mse_plot_width,
            height=cfg.analysis.mse_plot_height,
        )

    # cons_df, true_cons_df = _get_cons_dfs(cfg)
    # cons_df_indomain = cons_df.loc[np.isclose(cons_df.param, 0.6)]
    # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 0.6)]

    # cons_plot = (
    #     p9.ggplot(cons_df_indomain, p9.aes(x="t"))
    #     + p9.geom_line(p9.aes(y="lhs", color="model"))
    #     + p9.geom_line(p9.aes(y="rhs", color="model"))
    #     + p9.geom_line(p9.aes(y="true"), true_cons_df_indomain, linetype="dashed")
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     # + p9.facet_grid("~param")
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(strip_text=p9.element_text(usetex=True))
    #     + p9.ylab("Mass")
    # )

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    true_cons_df["model"] = "True"
    true_cons_df["trap"] = true_cons_df["true"]
    cons_df = pd.concat([cons_df, true_cons_df])
    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "True"]
    cons_df_indomain = (
        cons_df.loc[np.isclose(cons_df.param, 0.6)]
        .loc[lambda d: np.isin(d.model, models)]
        .assign(model=lambda d: pd.Categorical(d.model.astype(str), models))
        .assign(t_idx=lambda df: (df.t * 2000).astype(int))
        .groupby(["t_idx", "model"])
        .agg(
            trap_sd=pd.NamedAgg("trap", np.std),
            trap=pd.NamedAgg("trap", np.mean),
            true=pd.NamedAgg("true", np.mean),
        )
        .reset_index()
        .assign(t=lambda df: df.t_idx / 2000)
        .assign(trap_se=lambda d: d.trap_sd / np.sqrt(50))
        .assign(error=lambda d: d.trap - d.true)
    )
    # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 1)]
    colors = cfg.analysis.colors + ["#000000"]
    cons_df_indomain.loc[lambda d: np.isclose(d.t, 0.05)].to_csv(plot_dir / "stefan_cons.csv")

    cons_plot = (
        p9.ggplot(cons_df_indomain, p9.aes(x="t"))
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.geom_point(
            p9.aes(y="trap", color="model", shape="model"),
            cons_df_indomain.loc[lambda d: d.t_idx % 20 == 0],
            size=3,
        )
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        # + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.scale_color_manual(colors)
        # + p9.scale_fill_manual(colors.values())
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.ylab("Mass")
        + p9.labs(color="Model", shape="Model")
    )
    for ext in ("pdf", "png"):
        cons_plot.save(
            plot_dir / f"stefan_cons_plot.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.cons_plot_width * 1.5,
            height=cfg.analysis.cons_plot_height * 1.5,
        )

    # gmse = pw.load_ggplot(mse_at_t_plot, figsize=(4, 2))
    # gcons = pw.load_ggplot(cons_plot, figsize=(4, 2))
    # gboth = gmse / gcons
    # for ext in ("pdf", "png"):
    #     gboth.savefig(plot_dir / f"stefan_cons_mse_plot.{ext}")
    # cdi = cons_df_indomain.assign(tt=lambda df: (df.t * 2000).astype(int))
    # tcdi = true_cons_df_indomain.assign(tt=lambda df: (df.t * 2000).astype(int))
    # cdi = pd.merge(cdi, tcdi, on="tt")
    # cons_error_plot = (
    #     p9.ggplot(cdi, p9.aes(x="t_x"))
    #     + p9.geom_smooth(
    #         p9.aes(x="t_x", y="((lhs+rhs)/2 - true)**2", color="model"), se=False, span=0.3
    #     )
    #     # + p9.geom_smooth(p9.aes(x="t_x", y="(rhs - true)**2", color="model"), se=False, span=0.3)
    #     + p9.geom_smooth(
    #         p9.aes(x="t", y="MSE", color="model"),
    #         mse_at_t_df_indomain,
    #         se=False,
    #         span=0.05,
    #         linetype="dashed",
    #     )
    #     # + p9.geom_line(p9.aes(y="true - true"), linetype="dashed")
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     + p9.scale_y_log10()
    #     # + p9.facet_grid("~param")
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(strip_text=p9.element_text(usetex=True))
    #     + p9.ylab("Global Conservation Error")
    # )
    # for ext in ("pdf", "png"):
    #     cons_error_plot.save(
    #         plot_dir / f"stefan_cons_error_plot.{ext}",
    #         dpi=cfg.analysis.dpi,
    #         width=cfg.analysis.cons_plot_width,
    #         height=cfg.analysis.cons_plot_height,
    #     )


def linear_advection_plots():
    cfg = get_cfg_for_experiment("4b_advection_var_a")
    plot_dir = Path(cfg.base_dir).parent / "paper_plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    nice_names = cfg.analysis.nice_names
    colors = cfg.analysis.colors
    t = cfg.analysis.t_of_interest[0]
    fid = cfg.analysis.fids_of_interest[0]

    def labeller(x):
        if isinstance(x, float):
            return f"a = {int(x)}"
        elif x in ("1.0", "3.0"):
            return f"$\\beta$ = {int(float(x))}"
        else:
            return x

    shock_df = _make_shock_df(cfg, param_name="a_vals")
    a = 1
    # shock_df_in_domain = shock_df.loc[np.isclose(shock_df.pstar, a)]
    true_shock_value_in_domain = [0.5 + t, 0.5 + 3 * t]
    true_shock_df = pd.DataFrame(
        {
            "true_shock": true_shock_value_in_domain,
            "model_type": ["True shock", "True shock"],
            "pstar": [1.0, 3.0],
        }
    )
    true_shock_df["param"] = true_shock_df.pstar
    shock_plot_in_domain = (
        p9.ggplot(shock_df, p9.aes(x="shock_position"))  # noqa: WPS221
        + p9.geom_histogram(p9.aes(color="model", fill="model"), bins=15)
        + p9.geom_vline(
            p9.aes(xintercept="true_shock", linetype="model_type"), true_shock_df, size=1
        )
        # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=50, alpha=0.2)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["solid"])
        + p9.scale_x_continuous(limits=(0.3, 1.0))
        # + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
        + p9.facet_grid("pstar~model", labeller=labeller)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, 0.0),
            # axis_text_x=p9.element_text(size=9)
            # legend_position="top",
            legend_title=p9.element_blank(),
            # axis_title_y=p9.element_blank(),
        )
        + p9.xlab("Shock position")
        + p9.ylab("Count")
        + p9.labs(linetype="")
        + p9.guides(color=None, fill=None)
    )
    for ext in ("pdf", "png"):
        shock_plot_in_domain.save(
            plot_dir / f"advection_shock_plot_={fid}_t={t:.3f}.{ext}",
            # width=cfg.analysis.shock_plot_width,
            # height=cfg.analysis.shock_plot_height,
            width=cfg.analysis.time_plot_width,
            height=cfg.analysis.time_plot_height,
            dpi=cfg.analysis.dpi,
        )

    # Conservation / MSE (in-domain)
    plot_df = _get_plot_df(cfg)
    true_df = _get_true_df(cfg)
    fid = cfg.analysis.fids_of_interest[0]
    t_of_interest = cfg.analysis.t_of_interest
    assert len(t_of_interest) == 1, "only look at one time point"
    for df in (plot_df, true_df):
        df["t_of_interest"] = False
        df["t_label"] = ""
        for t in t_of_interest:
            df.loc[np.isclose(df.t, t), "t_of_interest"] = True
            df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
    plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
    true_df_at_ts = true_df.loc[true_df.t_of_interest]

    plot_df_f_id = plot_df_at_ts.loc[plot_df_at_ts.f_id == fid]
    true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == fid]

    # plot_df_f_id_indomain = plot_df_f_id.loc[np.isclose(plot_df_f_id.param, a)].assign(
    plot_df_f_id_indomain = plot_df_f_id.assign(x_idx=lambda d: (d.x * 200).astype(int))
    # true_df_f_id_indomain = true_df_f_id.loc[np.isclose(true_df_f_id.param, a)].assign(
    true_df_f_id_indomain = true_df_f_id.assign(x_idx=lambda d: (d.x * 200).astype(int))
    true_df_f_id_indomain["model_type"] = "True solution"
    plot_at_ts_indomain = (
        p9.ggplot(plot_df_f_id_indomain, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
        + p9.geom_point(
            p9.aes(color="model", group="model", shape="model"),
            plot_df_f_id_indomain.loc[lambda d: d.x_idx % 5 == 0],
            size=3,
        )
        + p9.geom_line(
            p9.aes(linetype="model_type"), data=true_df_f_id_indomain, color="black", size=1
        )
        # + p9.geom_vline(
        #     p9.aes(xintercept="true_shock"), true_shock_df, size=1, linetype="solid",
        # )
        # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=1, alpha=0.2)
        # + p9.scale_x_continuous(limits=(0.1, 0.35))
        + p9.scale_x_continuous(limits=(0.3, 1.0))
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["dashed"])
        # + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, 0.0),
            # axis_text_x=p9.element_text(size=9),
        )
        + p9.guides(color=None, fill=None, shape=None)
        + p9.labs(linetype="")
    )
    plot_at_ts_indomain_facet = (
        plot_at_ts_indomain
        + p9.facet_grid("param~model", labeller=labeller)
        + p9.geom_ribbon(
            p9.aes(
                ymin="u - 3 * u_sd", ymax="u + 3 * u_sd", color="model", fill="model", group="model"
            ),
            alpha=0.1,
        )
    )
    plot_at_ts_indomain_facet.save(
        plot_dir / f"advection_time_plot_indomain_fid={fid}_t={t:.2f}.pdf",
        width=cfg.analysis.time_plot_width,
        height=cfg.analysis.time_plot_height,
        dpi=cfg.analysis.dpi,
    )

    error_df_f_id = (
        plot_df_f_id_indomain.set_index(["param", "f_id", "x", "t"])
        .join(
            true_df_f_id_indomain.set_index(["param", "f_id", "x", "t"]),
            on=["param", "f_id", "x", "t"],
            rsuffix="_true",
        )
        .reset_index()
    )
    error_plot_at_ts = (
        p9.ggplot(error_df_f_id, mapping=p9.aes(x="x", y="u - u_true"))
        + p9.geom_line(p9.aes(color="model", group="model"))
        + p9.geom_point(p9.aes(color="model", group="model"))
        + p9.geom_ribbon(
            p9.aes(ymin="-3 * u_sd", ymax="3 * u_sd", color="model", fill="model", group="model"),
            alpha=0.1,
            linetype="dashed",
        )
        # + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed", size=1.5)
        + p9.facet_wrap("~model")
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.guides(color=None, fill=None)
        + p9.xlab("x")
        + p9.ylab("Error")
    )
    error_plot_at_ts.save(
        plot_dir / f"advection_error_time_plot.pdf",
        width=cfg.analysis.time_plot_width,
        height=cfg.analysis.time_plot_height,
        dpi=cfg.analysis.dpi,
    )

    mse_at_t_df = _get_mse_at_t_df(cfg)
    mse_at_t_df_indomain = mse_at_t_df.loc[np.isclose(mse_at_t_df.param, 0.6)]
    mse_at_t_plot = (
        p9.ggplot(mse_at_t_df_indomain, p9.aes(x="t", y="MSE", color="model"))  # noqa: WPS221
        + p9.geom_smooth(span=0.05, se=False)
        # + p9.facet_grid("~param")
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        # + p9.scale_color_hue(labels=nice_names.values())
        # + p9.guides(color=None)
        # + p9.theme(
        #     strip_text=p9.element_text(usetex=True), legend_position="bottom", legend_title=None
        # )
    )
    for ext in ("pdf", "png"):
        mse_at_t_plot.save(
            plot_dir / f"advection_mse_over_time.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.mse_plot_width,
            height=cfg.analysis.mse_plot_height,
        )

    # cons_df, true_cons_df = _get_cons_dfs(cfg)
    # cons_df_indomain = cons_df.loc[np.isclose(cons_df.param, 0.6)]
    # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 0.6)]

    # cons_plot = (
    #     p9.ggplot(cons_df_indomain, p9.aes(x="t"))
    #     + p9.geom_line(p9.aes(y="lhs", color="model"))
    #     + p9.geom_line(p9.aes(y="rhs", color="model"))
    #     + p9.geom_line(p9.aes(y="true"), true_cons_df_indomain, linetype="dashed")
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     # + p9.facet_grid("~param")
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(strip_text=p9.element_text(usetex=True))
    #     + p9.ylab("Mass")
    # )

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    true_cons_df["model"] = "True"
    true_cons_df["trap"] = true_cons_df["true"]
    cons_df = pd.concat([cons_df, true_cons_df])
    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "True"]
    cons_df_indomain = (
        # cons_df.loc[np.isclose(cons_df.param, a)]
        cons_df.loc[lambda d: np.isin(d.model, models)]
        .assign(model=lambda d: pd.Categorical(d.model.astype(str), models))
        .assign(t_idx=lambda df: np.round(df.t * 1000).astype(int))
        .groupby(["t_idx", "model", "param"])
        .agg(
            trap_sd=pd.NamedAgg("trap", np.std),
            trap=pd.NamedAgg("trap", np.mean),
            true=pd.NamedAgg("true", np.mean),
        )
        .reset_index()
        .assign(t=lambda df: df.t_idx / 1000)
        .assign(trap_se=lambda d: d.trap_sd / np.sqrt(50))
        .assign(error=lambda d: d.trap - d.true)
    )
    # cons_df_indomain.loc[lambda d: d.model=="True"].assign(trap = lambda d: 0.5 + d.t * d.param)
    # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 1)]
    colors = cfg.analysis.colors + ["#000000"]
    cons_df_indomain.loc[lambda d: np.isclose(d.t, 0.05)].to_csv(plot_dir / "advection_cons.csv")

    cons_plot = (
        p9.ggplot(cons_df_indomain, p9.aes(x="t"))
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.geom_point(
            p9.aes(y="trap", color="model", shape="model"),
            cons_df_indomain.loc[lambda d: d.t_idx % 100 == 0],
            size=3,
        )
        + p9.facet_wrap("~param", labeller=labeller)
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        # + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.scale_color_manual(colors)
        + p9.scale_x_continuous(limits=(0, 0.6))
        # + p9.scale_fill_manual(colors.values())
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.ylab("Mass")
        + p9.labs(color="Model", shape="Model")
    )
    for ext in ("pdf", "png"):
        cons_plot.save(
            plot_dir / f"advection_cons_plot.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.cons_plot_width * 1.5,
            height=cfg.analysis.cons_plot_height * 1.5,
        )
    print("here")


def burgers_plots():
    cfg = get_cfg_for_experiment("5b_burgers_var_a")
    plot_dir = Path(cfg.base_dir).parent / "paper_plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    nice_names = cfg.analysis.nice_names
    colors = cfg.analysis.colors
    t = cfg.analysis.t_of_interest[0]
    fid = cfg.analysis.fids_of_interest[0]

    def labeller(x):
        if isinstance(x, float):
            return f"a = {int(x)}"
        elif x in ("1.0", "3.0"):
            return f"$a$ = {int(float(x))}"
        else:
            return x

    # shock_df = _make_shock_df(cfg, param_name="a_vals")
    # a = 1
    # # shock_df_in_domain = shock_df.loc[np.isclose(shock_df.pstar, a)]
    # true_shock_value_in_domain = [0.5 + t, 0.5 + 3 * t]
    # true_shock_df = pd.DataFrame(
    #     {
    #         "true_shock": true_shock_value_in_domain,
    #         "model_type": ["True shock", "True shock"],
    #         "pstar": [1.0, 3.0],
    #     }
    # )
    # true_shock_df["param"] = true_shock_df.pstar
    # shock_plot_in_domain = (
    #     p9.ggplot(shock_df, p9.aes(x="shock_position"))  # noqa: WPS221
    #     + p9.geom_histogram(p9.aes(color="model", fill="model"), bins=15)
    #     + p9.geom_vline(
    #         p9.aes(xintercept="true_shock", linetype="model_type"), true_shock_df, size=1
    #     )
    #     # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=50, alpha=0.2)
    #     + p9.scale_color_manual(colors)
    #     + p9.scale_fill_manual(colors)
    #     + p9.scale_linetype_manual(["solid"])
    #     + p9.scale_x_continuous(limits=(0.3, 1.0))
    #     # + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
    #     + p9.facet_grid("pstar~model", labeller=labeller)
    #     + p9.theme_bw(base_size=cfg.analysis.base_font_size)
    #     + p9.theme(
    #         strip_text=p9.element_text(usetex=True),
    #         legend_position=(0.8, 0.0),
    #         # axis_text_x=p9.element_text(size=9)
    #         # legend_position="top",
    #         legend_title=p9.element_blank(),
    #         # axis_title_y=p9.element_blank(),
    #     )
    #     + p9.xlab("Shock position")
    #     + p9.ylab("Count")
    #     + p9.labs(linetype="")
    #     + p9.guides(color=None, fill=None)
    # )
    # for ext in ("pdf", "png"):
    #     shock_plot_in_domain.save(
    #         plot_dir / f"advection_shock_plot_={fid}_t={t:.3f}.{ext}",
    #         # width=cfg.analysis.shock_plot_width,
    #         # height=cfg.analysis.shock_plot_height,
    #         width=cfg.analysis.time_plot_width,
    #         height=cfg.analysis.time_plot_height,
    #         dpi=cfg.analysis.dpi,
    #     )

    plot_df = _get_plot_df(cfg)
    true_df = _get_true_df(cfg)
    fid = cfg.analysis.fids_of_interest[0]
    t_of_interest = cfg.analysis.t_of_interest
    assert len(t_of_interest) == 1, "only look at one time point"
    for df in (plot_df, true_df):
        df["t_of_interest"] = False
        df["t_label"] = ""
        for t in t_of_interest:
            df.loc[np.isclose(df.t, t), "t_of_interest"] = True
            df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
    plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
    true_df_at_ts = true_df.loc[true_df.t_of_interest]

    plot_df_f_id = plot_df_at_ts.loc[plot_df_at_ts.f_id == fid]
    true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == fid]

    # plot_df_f_id_indomain = plot_df_f_id.loc[np.isclose(plot_df_f_id.param, a)].assign(
    plot_df_f_id_indomain = plot_df_f_id.assign(x_idx=lambda d: np.round(d.x * 200).astype(int))
    # true_df_f_id_indomain = true_df_f_id.loc[np.isclose(true_df_f_id.param, a)].assign(
    true_df_f_id_indomain = true_df_f_id.assign(x_idx=lambda d: np.round(d.x * 200).astype(int))
    true_df_f_id_indomain["model_type"] = "True solution"
    plot_at_ts_indomain = (
        p9.ggplot(plot_df_f_id_indomain, mapping=p9.aes(x="x", y="u"))
        + p9.geom_line(p9.aes(color="model", group="model"), size=1.5)
        + p9.geom_point(
            p9.aes(color="model", group="model", shape="model"),
            plot_df_f_id_indomain.loc[lambda d: d.x_idx % 25 == 0],
            size=3,
        )
        + p9.geom_line(
            p9.aes(linetype="model_type"), data=true_df_f_id_indomain, color="black", size=1
        )
        # + p9.geom_vline(
        #     p9.aes(xintercept="true_shock"), true_shock_df, size=1, linetype="solid",
        # )
        # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=1, alpha=0.2)
        # + p9.scale_x_continuous(limits=(0.1, 0.35))
        + p9.scale_x_continuous(limits=(-1.0, 1.0))
        + p9.scale_color_manual(colors)
        + p9.scale_fill_manual(colors)
        + p9.scale_linetype_manual(["dashed"])
        # + p9.scale_x_continuous(limits=(0.1, 0.4), labels=["0.1", "0.2", "0.3", ""])
        # + p9.scale_x_continuous(limits=(0, 0.5))
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(
            strip_text=p9.element_text(usetex=True),
            legend_position=(0.8, 0.0),
            # axis_text_x=p9.element_text(size=9),
        )
        + p9.guides(color=None, fill=None, shape=None)
        + p9.labs(linetype="")
    )
    plot_at_ts_indomain_facet = (
        plot_at_ts_indomain
        + p9.facet_grid("param~model", labeller=labeller)
        + p9.geom_ribbon(
            p9.aes(
                ymin="u - 3 * u_sd", ymax="u + 3 * u_sd", color="model", fill="model", group="model"
            ),
            alpha=0.1,
        )
    )
    plot_at_ts_indomain_facet.save(
        plot_dir / f"burgers_time_plot_indomain_fid={fid}_t={t:.2f}.pdf",
        width=cfg.analysis.time_plot_width,
        height=cfg.analysis.time_plot_height,
        dpi=cfg.analysis.dpi,
    )

    cons_df, true_cons_df = _get_cons_dfs(cfg)
    true_cons_df["model"] = "True"
    true_cons_df["trap"] = true_cons_df["true"]
    cons_df = pd.concat([cons_df, true_cons_df])
    models = ["ANP", "SoftC-ANP", "HardC-ANP", "ProbConserv-ANP", "True"]
    cons_df_indomain = (
        # cons_df.loc[np.isclose(cons_df.param, a)]
        cons_df.loc[lambda d: np.isin(d.model, models)]
        .assign(model=lambda d: pd.Categorical(d.model.astype(str), models))
        .assign(t_idx=lambda df: np.round(df.t * 1000).astype(int))
        .groupby(["t_idx", "model", "param"])
        .agg(
            trap_sd=pd.NamedAgg("trap", np.std),
            trap=pd.NamedAgg("trap", np.mean),
            true=pd.NamedAgg("true", np.mean),
        )
        .reset_index()
        .assign(t=lambda df: df.t_idx / 1000)
        .assign(trap_se=lambda d: d.trap_sd / np.sqrt(50))
        .assign(error=lambda d: d.trap - d.true)
    )
    # cons_df_indomain.loc[lambda d: d.model=="True"].assign(trap = lambda d: 0.5 + d.t * d.param)
    # true_cons_df_indomain = true_cons_df.loc[np.isclose(true_cons_df.param, 1)]
    colors = cfg.analysis.colors + ["#000000"]
    cons_df_indomain.loc[lambda d: np.isclose(d.t, 0.05)].to_csv(plot_dir / "burgers_cons.csv")

    cons_plot = (
        p9.ggplot(cons_df_indomain, p9.aes(x="t"))
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.geom_point(
            p9.aes(y="trap", color="model", shape="model"),
            cons_df_indomain.loc[lambda d: d.t_idx % 100 == 0],
            size=3,
        )
        + p9.facet_wrap("~param", labeller=labeller)
        # + p9.geom_line(p9.aes(y="true"), linetype="dashed", size=1)
        # + p9.geom_line(p9.aes(y="trap", color="model"))
        + p9.scale_color_manual(colors)
        + p9.scale_x_continuous(limits=(0, 1.0))
        # + p9.scale_fill_manual(colors.values())
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.ylab("Mass")
        + p9.labs(color="Model", shape="Model")
    )
    for ext in ("pdf", "png"):
        cons_plot.save(
            plot_dir / f"burgers_cons_plot.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.cons_plot_width * 1.5,
            height=cfg.analysis.cons_plot_height * 1.5,
        )
    print("here")


def get_cfg_for_experiment(experiment: str):
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        overrides = [
            f"+experiments={experiment}",
        ]
        return compose(config_name="config", overrides=overrides)


def _make_shock_df(cfg, param_name="p_stars"):
    inference_results = torch.load(cfg.analysis.inference_results)
    shocks_all: Tensor = inference_results["shocks_all"]

    params = cfg.datasets.test[param_name]
    params_ordered = cfg.analysis.get("params_ordered", params)
    nice_names = cfg.analysis.nice_names
    t_range = cfg.analysis.t_range
    t_of_interest = cfg.analysis.t_of_interest
    nx = cfg.analysis.nx
    fids_of_interest = cfg.analysis.fids_of_interest
    n_shock_samples = cfg.analysis.n_shock_samples

    # Shock position
    t = t_of_interest[0]
    fid = fids_of_interest[0]
    t_idx = int((t / t_range[1]) * (nx - 1))
    shocks_at_t_and_fid = shocks_all[:, :, :, int(fid), t_idx]
    shocks_at_t_and_fid = rearrange(shocks_at_t_and_fid, "nm nsamples nd -> (nm nsamples nd) 1")
    midx = pd.MultiIndex.from_product(
        [nice_names.values(), range(n_shock_samples), params],
        names=["model", "sample", "pstar"],
    )
    out_df = (
        pd.DataFrame(shocks_at_t_and_fid, index=midx, columns=["shock_position"])
        .reset_index()
        .assign(pstar=lambda df: pd.Categorical(df.pstar, params_ordered, ordered=True))
        .assign(model=lambda df: pd.Categorical(df.model, nice_names.values(), ordered=True))
    )
    return out_df


def _get_true_shock_value(cfg, t: float, pstar: float):
    nx = cfg.analysis.nx
    t_range = cfg.analysis.t_range
    t_idx = int((t / t_range[1]) * (nx - 1))
    true_stefan = Stefan(pstar)
    return true_stefan.alpha * np.sqrt(t_idx / nx * 0.1)


def _get_shock_range(cfg):
    t = cfg.analysis.t_of_interest[0]
    min_shock = _get_true_shock_value(cfg, t, cfg.datasets.p_star_max)
    max_shock = _get_true_shock_value(cfg, t, cfg.datasets.p_star_min)
    return min_shock, max_shock


def _get_plot_df(cfg):
    df = pd.read_pickle(cfg.analysis.plot_df_path)
    nice_names = cfg.analysis.nice_names
    c = pd.Categorical(df["model"], categories=nice_names.keys(), ordered=True).rename_categories(
        nice_names
    )
    df["model"] = c
    return df


def _get_true_df(cfg):
    return pd.read_pickle(cfg.analysis.true_df_path)


def _get_mse_at_t_df(cfg):
    df = pd.read_pickle(cfg.analysis.mse_at_t_df_path)
    nice_names = cfg.analysis.nice_names
    c = pd.Categorical(df["model"], categories=nice_names.keys(), ordered=True).rename_categories(
        nice_names
    )
    df["model"] = c
    return df


def _get_cons_dfs(cfg):
    cons_df = pd.read_pickle(cfg.analysis.cons_df_path)
    nice_names = cfg.analysis.nice_names
    c = pd.Categorical(
        cons_df["model"], categories=nice_names.keys(), ordered=True
    ).rename_categories(nice_names)
    cons_df["model"] = c
    true_cons_df = pd.read_pickle(cfg.analysis.true_cons_df_path)
    return cons_df, true_cons_df


if __name__ == "__main__":
    main()
