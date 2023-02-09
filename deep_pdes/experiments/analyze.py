from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra
import numpy as np
import pandas as pd
import plotnine as p9
import torch
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from torch import Tensor
from torch.nn import functional as F  # noqa: WPS347
from tqdm import tqdm

from deep_pdes.attentive_neural_process.anp import ANP
from deep_pdes.attentive_neural_process.probconserv import (
    InequalityConstraint,
    PhysNP,
    apply_non_linear_ineq_constraint,
)
from deep_pdes.attentive_neural_process.softc import PINP, SoftcANP
from deep_pdes.datasets.base import ANPDataset
from deep_pdes.datasets.pme import (
    Burgers,
    HeatEquation,
    LinearAdvection,
    PorousMediumEquation,
    Stefan,
    StefanPME,
    mass_burgers,
    mass_pme,
    meshgrid,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    preds = infer(cfg)

    dataset_path = Path(cfg.datasets.save_path)
    test_load_path = dataset_path / "test.pt"
    test_dataset = instantiate(cfg.datasets.test, load_path=test_load_path)
    _, output_targets, _ = get_test_solution(cfg, test_dataset)
    mse_at_t_df = analyze_mean_squared_error(cfg, preds, output_targets, test_dataset)
    mse_at_t_df.to_pickle(cfg.analysis.mse_at_t_df_path)

    plot_df, true_df = make_plotting_dfs(cfg, preds, output_targets, test_dataset)
    plot_df.to_pickle(cfg.analysis.plot_df_path)
    true_df.to_pickle(cfg.analysis.true_df_path)

    cons_df, true_cons_df = analyze_conservation(cfg, test_dataset, plot_df)
    cons_df.to_pickle(cfg.analysis.cons_df_path)
    true_cons_df.to_pickle(cfg.analysis.true_cons_df_path)

    if (
        isinstance(test_dataset, StefanPME)
        or isinstance(test_dataset, PorousMediumEquation)
        or isinstance(test_dataset, LinearAdvection)
    ):
        shocks_all = analyze_shocks(cfg, preds, test_dataset)
    else:
        shocks_all = None
    plot_context_points(cfg, test_dataset, plot_df, true_df)
    analyze_solution_profiles(cfg, test_dataset, plot_df, true_df)

    inference_results = {
        "shocks_all": shocks_all,
    }
    torch.save(inference_results, cfg.analysis.inference_results)


def analyze_mean_squared_error(
    cfg, preds: Dict[str, Tensor], output_targets: Tensor, test_dataset: ANPDataset
) -> pd.DataFrame:
    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir()
    t_range = cfg.analysis.t_range
    params = get_params(cfg, test_dataset)
    n_p_stars = len(params)
    params_ordered = cfg.analysis.get("params_ordered", params)
    nice_names = cfg.analysis.nice_names
    models_of_interest: List[str] = list(nice_names.keys())

    all_preds = torch.stack([p["pred"] for p in preds.values()])
    error = all_preds - rearrange(output_targets, "nf nx_nt -> 1 nf nx_nt")
    squared_error = error.pow(2)

    mse_by_t_and_fid = reduce(
        squared_error,
        "nm (nd nfpd) (nx nt) -> nm nd nfpd nt",
        "mean",
        nd=n_p_stars,
        nx=cfg.analysis.nx,
    )
    mse_by_t = reduce(mse_by_t_and_fid, "nm nd nfpd nt -> nm nd nt", "mean", nd=n_p_stars)
    mse_sd_by_t = reduce(
        mse_by_t_and_fid,
        "nm nd nfpd nt -> nm nd nt",
        torch.std,
        nd=n_p_stars,
    )
    nfpd = squared_error.shape[1] / n_p_stars
    mse_se_by_t = mse_sd_by_t / torch.tensor(nfpd).sqrt()
    ts = np.linspace(*t_range, cfg.analysis.nt)
    dim_values = [preds.keys(), params, ts]
    dim_names = ["model", "param", "t"]
    midx = pd.MultiIndex.from_product(dim_values, names=dim_names)
    mse_all_by_t = torch.cat(
        [mse_by_t.reshape(-1, 1), mse_sd_by_t.reshape(-1, 1), mse_se_by_t.reshape(-1, 1)], dim=-1
    )
    mse_at_t_long_df = pd.DataFrame(
        mse_all_by_t, index=midx, columns=["MSE", "MSE_sd", "MSE_se"]
    ).reset_index()

    # scales = torch.stack([p["scale"].reshape(p["scale"].shape[0], -1) if p["scale"] is not None else torch.tensor(torch.nan).expand(p["pred"].shape) for p in preds.values()])
    def _get_scale(model, preds):
        if "physnp" in model:
            # loc = preds[model]["loc"]
            # cov = preds["physnp_notrain"]["cov"]
            scale = rearrange(preds["scale"], "nf nt nx 1 -> nf (nx nt)", nx=cfg.analysis.nx)
        elif ("anp" in model) or ("pinp" in model):
            # scale = rearrange(preds[model]["scale"], "nf (nx nt) 1 -> nf nt nx 1", nx=cfg.analysis.nx)
            scale = rearrange(preds["scale"], "nf (nx nt) 1 -> nf (nx nt)", nx=cfg.analysis.nx)
        elif "hcnp" in model:
            scale = rearrange(preds["scale"], "nf nt nx 1 -> nf (nx nt)", nx=cfg.analysis.nx)
        else:
            scale = torch.tensor(torch.nan).expand(preds["pred"].shape)
        return scale

    scales = torch.stack([_get_scale(k, v) for (k, v) in preds.items()])
    loglik = -0.5 * squared_error / scales.pow(2) - scales.log() - 0.5 * np.log(2 * torch.pi)
    loglik_by_t_and_fid = reduce(
        loglik, "nm (nd nfpd) (nx nt) -> nm nd nfpd nt", "mean", nd=n_p_stars, nx=cfg.analysis.nx
    )
    loglik_by_t = reduce(loglik_by_t_and_fid, "nm nd nfpd nt -> nm nd nt", "mean")
    loglik_sd_by_t = reduce(loglik_by_t_and_fid, "nm nd nfpd nt -> nm nd nt", torch.std)
    loglik_se_by_t = loglik_sd_by_t / torch.tensor(nfpd).sqrt()

    loglik_at_fid = rearrange(
        loglik, "nm (nd nfpd) (nx nt) -> nm nd nfpd nx nt", nd=n_p_stars, nx=cfg.analysis.nx
    )[:, :, 1]
    loglik_by_t_at_fid = reduce(loglik_at_fid, "nm nd nx nt -> nm nd nt", "mean")

    mse_at_t_long_df["loglik"] = loglik_by_t.reshape(-1, 1)
    mse_at_t_long_df["loglik_sd"] = loglik_sd_by_t.reshape(-1, 1)
    mse_at_t_long_df["loglik_se"] = loglik_se_by_t.reshape(-1, 1)
    mse_at_t_long_df["loglik_fid"] = loglik_by_t_at_fid.reshape(-1, 1)
    mse_at_t_long_df = (
        mse_at_t_long_df.loc[np.isin(mse_at_t_long_df.model, models_of_interest)]
        .assign(model=lambda df: pd.Categorical(df.model, models_of_interest, ordered=True))
        .assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))
    )

    t_of_interest = cfg.analysis.t_of_interest[0]
    m: pd.DataFrame = mse_at_t_long_df.loc[np.isclose(mse_at_t_long_df.t, t_of_interest)]
    m.to_csv(outdir / "mse_at_t.csv")

    labeller = make_labeller(cfg)
    mse_at_t_plot = (
        p9.ggplot(mse_at_t_long_df, p9.aes(x="t", y="MSE", color="model"))  # noqa: WPS221
        + p9.geom_smooth(span=0.05, se=False)
        + p9.facet_grid("~param", labeller=labeller)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.scale_color_hue(labels=nice_names.values())
        # + p9.guides(color=None)
        + p9.theme(
            strip_text=p9.element_text(usetex=True), legend_position="bottom", legend_title=None
        )
    )

    plot_dir = outdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    for ext in ("pdf", "png"):
        mse_at_t_plot.save(
            plot_dir / f"mse_over_time.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.mse_plot_width,
            height=cfg.analysis.mse_plot_height,
        )

    return mse_at_t_long_df


def _get_loc_and_cov(model: str, preds: Dict[str, Dict[str, Tensor]], nx: int):
    if ("physnp_notrain" in model) or ("physnp_limit" in model) or ("physnp_second_deriv" in model):
        loc = preds[model]["loc"]
        cov = preds[model]["cov"]
    elif ("anp" in model) or ("pinp" in model):
        loc = rearrange(preds[model]["loc"], "nf (nx nt) 1 -> nf nt nx 1", nx=nx)
        scale = rearrange(preds[model]["scale"], "nf (nx nt) 1 -> nf nt nx 1", nx=nx)
        eye = rearrange(torch.eye(nx), "nx1 nx2 -> 1 1 nx1 nx2")
        cov = scale * eye
    elif "hcnp" in model:
        loc = rearrange(preds[model]["loc"], "nf nt nx 1 -> nf nt nx 1", nx=nx)
        scale = rearrange(preds[model]["scale"], "nf nt nx 1 -> nf nt nx 1", nx=nx)
        eye = rearrange(torch.eye(nx), "nx1 nx2 -> 1 1 nx1 nx2")
        cov = scale * eye
    else:
        loc = None
        cov = None
    return loc, cov


def analyze_shocks(cfg, preds: Dict[str, Tensor], test_dataset: ANPDataset) -> Tensor:
    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir()
    t_range = cfg.analysis.t_range
    if isinstance(test_dataset, PorousMediumEquation):
        params = cfg.datasets.test.degrees
        models = params  # shock is constant
    elif isinstance(test_dataset, StefanPME):
        params = cfg.datasets.test.p_stars
        models = [Stefan(p) for p in params]
    elif isinstance(test_dataset, LinearAdvection):
        params = cfg.datasets.test.a_vals
        models = params
    else:
        raise NotImplementedError()
    params_ordered = cfg.analysis.get("params_ordered", params)
    nice_names = cfg.analysis.nice_names
    x_range = cfg.analysis.x_range
    t_of_interest = cfg.analysis.t_of_interest
    nx = cfg.analysis.nx
    fids_of_interest = cfg.analysis.fids_of_interest
    shocks = {}
    n_shock_samples = cfg.analysis.n_shock_samples
    n_shock_samples_per_batch: Optional[int] = cfg.analysis.get("n_shock_samples_per_batch")
    for model in nice_names.keys():
        shock_path_str = cfg.analysis.methods[model].get("shock_path", None)
        shock_overwrite = cfg.analysis.methods[model].get("shock_overwrite", False)
        if shock_path_str is not None:
            shock_path = Path(shock_path_str)
        else:
            shock_path = None
        if (shock_path is None) or (not shock_path.exists()) or shock_overwrite:
            loc, cov = _get_loc_and_cov(model, preds, nx)
            if cov is None:
                continue
            s = estimate_shock_interval(
                loc.cuda(),
                cov.cuda(),
                n_samples=n_shock_samples,
                n_samples_per_batch=n_shock_samples_per_batch,
            )
            s = rearrange(s, "ns (nd nfpd) nt -> ns nd nfpd nt", nd=len(models))
            s = (s / nx) * (x_range[1])
            shocks[nice_names[model]] = s.cpu()
            if shock_path is not None:
                torch.save(s.cpu(), shock_path)
        else:
            shocks[nice_names[model]] = torch.load(shock_path)
    shocks_all: Tensor = torch.stack(tuple(shocks.values()))
    for fid in fids_of_interest:
        t = t_of_interest[0]
        t_idx = int((t / t_range[1]) * (nx - 1))
        dfs = []
        true_shock_dfs = []
        shocks_at_t_and_fid = shocks_all[:, :, :, int(fid), t_idx]
        shocks_at_t_and_fid = rearrange(shocks_at_t_and_fid, "nm nsamples nd -> (nm nsamples nd) 1")
        midx = pd.MultiIndex.from_product(
            [shocks.keys(), range(n_shock_samples), params], names=["model", "sample", "param"]
        )
        shock_df = (
            pd.DataFrame(shocks_at_t_and_fid, index=midx, columns=["shock_position"])
            .reset_index()
            .assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))
        )
        if isinstance(test_dataset, StefanPME):
            for i, model in enumerate(models):
                true_shock = model.alpha * np.sqrt(t_idx / nx * 0.1)
                min_stefan = Stefan(p_star=cfg.datasets.p_star_max)
                min_shock = min_stefan.alpha * np.sqrt(t_idx / nx * 0.1)
                max_stefan = Stefan(p_star=cfg.datasets.p_star_min)
                max_shock = max_stefan.alpha * np.sqrt(t_idx / nx * 0.1)
                true_shock_dfs.append(
                    pd.DataFrame(
                        {
                            "param": np.array([model.p_star]),
                            "true_shock": np.array([true_shock]),
                            "max_shock": np.array([max_shock]),
                        }
                    )
                )
        elif isinstance(test_dataset, PorousMediumEquation):
            for i, model in enumerate(models):
                true_shock = t
                true_shock_dfs.append(
                    pd.DataFrame({"param": np.array([model]), "true_shock": np.array([true_shock])})
                )
        elif isinstance(test_dataset, LinearAdvection):
            for i, param in enumerate(params):
                true_shock = 0.5 + t * param
                true_shock_dfs.append(
                    pd.DataFrame({"param": np.array([param]), "true_shock": np.array([true_shock])})
                )
        else:
            raise NotImplementedError()

        true_shock_df = pd.concat(true_shock_dfs, axis=0).assign(
            param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True)
        )
        labeller = make_labeller(cfg)
        shock_plot_i = (
            p9.ggplot(
                shock_df, p9.aes(x="shock_position", color="model", fill="model")
            )  # noqa: WPS221
            + p9.geom_histogram(bins=50)
            + p9.geom_vline(data=true_shock_df, mapping=p9.aes(xintercept="true_shock"))
            # + p9.annotate("rect", xmin=min_shock, xmax=max_shock, ymin=0, ymax=50, alpha=0.2)
            + p9.facet_grid("model~param", labeller=labeller)
            + p9.scale_color_hue(labels=nice_names.values())
            + p9.scale_fill_hue(labels=nice_names.values())
            + p9.theme_bw(base_size=cfg.analysis.base_font_size)
            + p9.theme(
                strip_text=p9.element_text(usetex=True),
                legend_position="top",
                legend_title=p9.element_blank(),
            )
            + p9.xlab("x")
        )
        plot_dir = outdir / "plots"
        if not plot_dir.exists():
            plot_dir.mkdir()
        for ext in ("pdf", "png"):
            shock_plot_i.save(
                plot_dir / f"shock_plot_fid={fid}_t={t:.3f}.{ext}",
                width=cfg.analysis.shock_plot_width,
                height=cfg.analysis.shock_plot_height,
                dpi=cfg.analysis.dpi,
            )
    return shocks_all
    # return {"shock_plot_i": shock_plot_i}


def get_params(cfg, test_dataset: ANPDataset):
    if isinstance(test_dataset, PorousMediumEquation):
        params = cfg.datasets.test.degrees
    elif isinstance(test_dataset, StefanPME):
        params = cfg.datasets.test.p_stars
    elif isinstance(test_dataset, HeatEquation):
        params = cfg.datasets.test.conductivities
    elif isinstance(test_dataset, LinearAdvection):
        params = cfg.datasets.test.a_vals
    elif isinstance(test_dataset, Burgers):
        params = cfg.datasets.test.a_vals
    else:
        raise NotImplementedError()
    return params


def make_plotting_dfs(
    cfg, preds: Dict[str, Tensor], output_targets: Tensor, test_dataset: ANPDataset
):
    plot_dfs = defaultdict(dict)
    params = get_params(cfg, test_dataset)
    n_p_stars = len(params)
    x_range = cfg.analysis.x_range
    t_range = cfg.analysis.t_range
    models_of_interest = list(cfg.analysis.nice_names.keys())
    params_ordered = cfg.analysis.get("params_ordered", params)

    # nf = next(iter(preds.values()))["pred"]
    nx = cfg.analysis.nx
    nt = cfg.analysis.nt
    pred_loc_all = torch.stack([p["pred"] for p in preds.values()])
    nf = pred_loc_all.shape[1]
    pred_sd_list = []
    for model, pred_dict in preds.items():
        scale = pred_dict.get("scale")
        if scale is None:
            pred_sd = torch.zeros((nf, nx * nt))
        elif len(scale.shape) == 3:
            pred_sd = rearrange(scale, "nf nx_nt 1 -> nf nx_nt")
        else:
            pred_sd = rearrange(scale, "nf nt nx 1 -> nf (nx nt)")
        pred_sd_list.append(pred_sd)
    pred_sd_all = torch.stack(pred_sd_list)
    pred_all = torch.stack((pred_loc_all, pred_sd_all), dim=-1)
    nfpd = nf // n_p_stars
    pred_all = rearrange(pred_all, "nm nf nx_nt d -> (nm nf nx_nt) d")
    xs = np.linspace(*x_range, nx)
    ts = np.linspace(*t_range, nt)
    midx = pd.MultiIndex.from_product(
        (preds.keys(), params, np.array(range(nfpd)).astype(str), xs, ts),
        names=("model", "param", "f_id", "x", "t"),
    )
    plot_df = (
        pd.DataFrame(pred_all, index=midx, columns=["u", "u_sd"])
        .reset_index()
        .assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))
    )
    plot_df = plot_df.loc[np.isin(plot_df.model, models_of_interest)].assign(
        model=lambda df: pd.Categorical(df.model, models_of_interest, ordered=True)
    )

    true_df = pd.DataFrame(output_targets)
    # true_df = pd.DataFrame(output_targets.unsqueeze(0))
    true_df["param"] = pd.Categorical(np.repeat(params, nfpd), params_ordered, ordered=True)
    true_df["f_id"] = np.tile(range(nfpd), n_p_stars)
    true_df["f_id"] = true_df["f_id"].astype(str)
    true_df = true_df.melt(id_vars=("param", "f_id"), value_name="u")
    true_df["x"] = np.repeat(
        np.repeat(np.linspace(x_range[0], x_range[1], cfg.analysis.nx), cfg.analysis.nt), nf
    )
    true_df["t"] = np.repeat(
        np.tile(np.linspace(t_range[0], t_range[1], cfg.analysis.nt), cfg.analysis.nx), nf
    )

    return plot_df, true_df


def analyze_solution_profiles(
    cfg, test_dataset: ANPDataset, plot_df: pd.DataFrame, true_df: pd.DataFrame
):
    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir()
    plot_dir = outdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()

    t_of_interest = cfg.analysis.t_of_interest
    params = get_params(cfg, test_dataset)
    n_p_stars = len(params)
    nf = len(test_dataset)
    nfpd = nf // n_p_stars
    n_models = len(cfg.analysis.nice_names)

    fids_of_interest = cfg.analysis.fids_of_interest
    params_ordered = cfg.analysis.get("params_ordered", params)

    if t_of_interest is not None:
        assert len(t_of_interest) == 1, "only look at one time point"
        for df in (plot_df, true_df):
            df["t_of_interest"] = False
            df["t_label"] = ""
            for t in t_of_interest:
                df.loc[np.isclose(df.t, t), "t_of_interest"] = True
                df.loc[np.isclose(df.t, t), "t_label"] = f"{t:.3f}"
        plot_df_at_ts = plot_df.loc[plot_df.t_of_interest]
        true_df_at_ts = true_df.loc[true_df.t_of_interest]

        labeller = make_labeller(cfg)
        plot_at_ts = (
            p9.ggplot(plot_df_at_ts, mapping=p9.aes(x="x", y="u", group="f_id"))
            + p9.geom_line(p9.aes(color="f_id"), alpha=0.6)
            + p9.geom_line(data=true_df_at_ts, color="blue", linetype="dashed")
            + p9.facet_grid("param~model", labeller=labeller)
            + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        )
        plot_at_ts.save(plot_dir / f"time_plot.pdf", width=8, height=8, dpi=cfg.analysis.dpi)
        if fids_of_interest is not None:
            for f_id in fids_of_interest:
                input_idxs = range(int(f_id), nf, nfpd)
                input_contexts = test_dataset.tensors["input_contexts"]
                ic = input_contexts[input_idxs][:, :, 1]
                ic = rearrange(ic, "nd (nt nx) -> nd nt nx", nx=test_dataset.n_contexts_x)[:, 0]
                ic_df = (
                    pd.DataFrame(ic.transpose(1, 0), columns=params)
                    .melt(var_name="param", value_name="x")
                    .assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))
                )
                plot_df_f_id = plot_df_at_ts.loc[plot_df_at_ts.f_id == f_id]
                true_df_f_id = true_df_at_ts.loc[true_df_at_ts.f_id == f_id]
                plot_at_ts_uncertainty = (
                    p9.ggplot(plot_df_f_id, mapping=p9.aes(x="x", y="u"))
                    + p9.geom_line(p9.aes(color="model"))
                    + p9.geom_ribbon(
                        p9.aes(
                            ymin="u - 3 * u_sd", ymax="u + 3 * u_sd", color="model", fill="model"
                        ),
                        alpha=0.2,
                    )
                    + p9.geom_line(data=true_df_f_id, color="black", linetype="dashed")
                    + p9.geom_point(
                        data=ic_df,
                        shape="x",
                        inherit_aes=False,
                        mapping=p9.aes(x="x", y=0),
                        alpha=0.7,
                    )
                    + p9.scale_color_hue(labels=["ANP", "PhysNP"], guide=None)
                    + p9.scale_fill_hue(labels=["ANP", "PhysNP"], guide=None)
                    + p9.facet_grid("model~param", labeller=labeller)
                    + p9.theme_bw(base_size=cfg.analysis.base_font_size)
                    + p9.theme(strip_text=p9.element_text(usetex=True))
                )
                plot_at_ts_uncertainty.save(
                    plot_dir / f"time_plot_fid={f_id}_t={t:.2f}.pdf",
                    width=cfg.analysis.time_plot_width,
                    height=cfg.analysis.time_plot_height,
                    dpi=cfg.analysis.dpi,
                )


def plot_context_points(
    cfg, test_dataset: ANPDataset, plot_df: pd.DataFrame, true_df: pd.DataFrame
):
    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir()
    plot_dir = outdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    ## Plot context points
    input_contexts = test_dataset.tensors["input_contexts"]

    params = get_params(cfg, test_dataset)
    n_p_stars = len(params)
    nf = len(test_dataset)
    nfpd = nf // n_p_stars
    f_ids_of_interest = cfg.analysis.get("fids_of_interest")
    params_ordered = cfg.analysis.get("params_ordered", params)

    x_range = cfg.analysis.x_range
    t_range = cfg.analysis.t_range
    if f_ids_of_interest is not None:
        for j, f_id in enumerate(f_ids_of_interest):
            input_idxs = range(int(f_id), nf, nfpd)
            ic = input_contexts[input_idxs]
            ic = rearrange(ic, "nd (nt nx) d -> (nd nt nx) d", nx=test_dataset.n_contexts_x)
            midx = pd.MultiIndex.from_product([params, range(100)], names=["param", "i"])
            ic_df = (
                pd.DataFrame(ic, index=midx, columns=["t", "x"])
                .reset_index()
                .assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))
            )
            labeller = make_labeller(cfg)
            input_context_param_fid_plot = (
                p9.ggplot(ic_df, mapping=p9.aes(x="x", y="t"))
                + p9.geom_point(shape="x")
                + p9.facet_grid("~param", labeller=labeller)
                + p9.theme_bw(base_size=cfg.analysis.base_font_size)
                + p9.theme(strip_text=p9.element_text(usetex=True))
                + p9.scale_x_continuous(limits=x_range)
                + p9.scale_y_continuous(limits=t_range)
            )
            input_context_param_fid_plot.save(
                plot_dir / f"input_context_fid={f_id}.pdf",
                width=10,
                height=6,
                dpi=cfg.analysis.dpi,
            )


def analyze_conservation(cfg, test_dataset: ANPDataset, plot_df: pd.DataFrame):
    t_range = cfg.analysis.t_range
    x_range = cfg.analysis.x_range
    domain_length = x_range[1] - x_range[0]
    params = get_params(cfg, test_dataset)
    n_p_stars = len(params)
    params_ordered = cfg.analysis.get("params_ordered", params)
    nice_names = cfg.analysis.nice_names

    true_mass = get_analytical_mass_rhs(
        test_dataset, 0, len(test_dataset), torch.linspace(*t_range, cfg.analysis.nt)
    )
    true_mass = rearrange(true_mass, "(nd nfpd) nt -> nd nfpd nt", nd=n_p_stars)[:, 0]
    true_cons_df = pd.DataFrame(true_mass.transpose(1, 0), columns=params)
    true_cons_df["t"] = torch.linspace(*t_range, cfg.analysis.nt)
    true_cons_df["t_idx"] = torch.arange(0, cfg.analysis.nt)
    true_cons_df = true_cons_df.melt(
        id_vars=["t", "t_idx"], var_name="param", value_name="true"
    ).assign(param=lambda df: pd.Categorical(df.param, params_ordered, ordered=True))

    cons_df = (
        plot_df.assign(
            t_idx=lambda df: (np.round((cfg.analysis.nt - 1) * (df.t / df.t.max()))).astype(int)
        )
        .groupby(by=["param", "model", "t_idx", "f_id"])
        .agg(
            lhs=pd.NamedAgg("u", lambda x: x[1:].mean() * domain_length),
            rhs=pd.NamedAgg("u", lambda x: x[:-1].mean() * domain_length),
        )
        .reset_index()
        .assign(trap=lambda df: (df["rhs"] + df["lhs"]) / 2)
        .set_index(["param", "t_idx"])
        .join(true_cons_df.set_index(["param", "t_idx"]))
        .reset_index()
        .assign(error=lambda df: df["trap"] - df["true"])
    )

    cons_df_for_plot = cons_df.loc[lambda df: df.f_id == "1"]

    labeller = make_labeller(cfg)
    cons_plot = (
        p9.ggplot(cons_df_for_plot, p9.aes(x="t"))
        + p9.geom_line(p9.aes(y="lhs", color="model"))
        + p9.geom_line(p9.aes(y="rhs", color="model"))
        + p9.geom_line(p9.aes(y="true"), true_cons_df, linetype="dashed")
        + p9.facet_grid("~param", labeller=labeller)
        + p9.theme_bw(base_size=cfg.analysis.base_font_size)
        + p9.theme(strip_text=p9.element_text(usetex=True))
        + p9.scale_color_hue(labels=nice_names.values())  # , guide=None)
        + p9.ylab("Mass")
    )

    outdir = Path(cfg.analysis.outdir)
    if not outdir.exists():
        outdir.mkdir()
    plot_dir = outdir / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir()
    for ext in ("pdf", "png"):
        cons_plot.save(
            plot_dir / f"cons_plot.{ext}",
            dpi=cfg.analysis.dpi,
            width=cfg.analysis.cons_plot_width,
            height=cfg.analysis.cons_plot_height,
        )

    # cons_df = cons_df.set_index(["param", "t"]).join(true_cons_df.set_index(["param", "t"])).reset_index().assign(error = lambda df: df["trap"] - df["true"])
    # cons_df_at_t = cons_df.loc[lambda df: np.isin(df["t"], cfg.analysis.t_of_interest[0])]
    cons_df_at_t = (
        cons_df.loc[lambda df: np.isclose(df.t, cfg.analysis.t_of_interest[0])]
        .groupby(by=["param", "model"])
        .agg(error=pd.NamedAgg("error", np.mean))
        .reset_index()
    )
    cons_df_at_t.to_csv(plot_dir / "cons_df.csv")

    return cons_df, true_cons_df


def make_labeller(cfg):
    nice_names = cfg.analysis.nice_names

    def labeller(value):
        if value in nice_names:
            return nice_names[value]
        elif value == "0.5":
            return "Outside training range~($u^\\star=0.5$)"
        elif value == "0.6":
            return "Inside training range~($u^\\star=0.6$)"
        else:
            return value

    return labeller


def infer(cfg) -> Dict[str, Dict[str, Tensor]]:
    dataset_path = Path(cfg.datasets.save_path)
    test_load_path = dataset_path / "test.pt"
    test_dataset = instantiate(cfg.datasets.test, load_path=test_load_path)

    input_targets, _, input_ts = get_test_solution(cfg, test_dataset)

    gpu = torch.device(cfg.analysis.gpu)

    out = {}

    for method_name, method_cfg in cfg.analysis.methods.items():
        if method_cfg.get("use_empirical_mass", False):
            mass_rhs_in = get_empirical_mass_rhs(cfg)
        else:
            mass_rhs_in = None
        out[method_name] = run_inference_for_method(
            method_name,
            method_cfg,
            test_dataset,
            input_targets,
            gpu,
            input_ts,
            mass_rhs_in=mass_rhs_in,
        )
        if method_cfg.get("truncated_version", False):
            out[f"{method_name}_trunc"] = truncate_results(out[method_name])
        if method_cfg.get("constrained_version", False):
            constrained_results_path = Path(method_cfg.infer_path_constrained)
            if (not constrained_results_path.exists()) or method_cfg.overwrite_constrained:
                cnstrd = constrained_results(out[method_name], "monotone")
                torch.save(cnstrd, constrained_results_path)
            else:
                cnstrd = torch.load(constrained_results_path)
            out[f"{method_name}_cnstrd"] = cnstrd
            out[f"{method_name}_cnstrd_trunc"] = truncate_results(cnstrd)

            if method_cfg.get("nonneg_path", False):
                nonneg_path = Path(method_cfg.nonneg_path)
                if (not nonneg_path.exists()) or method_cfg.overwrite_nonneg:
                    nonneg = constrained_results(out[method_name], "nonneg")
                    torch.save(nonneg, nonneg_path)
                else:
                    nonneg = torch.load(nonneg_path)
                out[f"{method_name}_nonneg"] = nonneg
                out[f"{method_name}_nonneg_trunc"] = truncate_results(nonneg)

    return out


Model = Union[ANP, PhysNP]


def run_inference_for_method(
    name: str, cfg, test_dataset: ANPDataset, input_targets, gpu, input_ts, mass_rhs_in=None
):

    seed: Optional[int] = cfg.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
    nt = input_ts.shape[0]
    save_path = Path(cfg.infer_path)
    if (not save_path.exists()) or cfg.overwrite:
        model: Model = instantiate(cfg.model)
        if cfg.state_dict is not None:
            state_dict = torch.load(cfg.state_dict)
            model.load_state_dict(state_dict)
        else:
            state_dict = model.state_dict()
        results = {}
        if isinstance(model, PhysNP):
            constraint_precision = cfg.constraint_precision
            if constraint_precision is not None:
                state_dict["log_constraint_precision_train"] = torch.tensor(
                    constraint_precision
                ).log()
                model.load_state_dict(state_dict)
            anp_state_dict_path = cfg.anp_state_dict
            if anp_state_dict_path is not None:
                anp_state_dict = torch.load(anp_state_dict_path)
                model.anp.load_state_dict(anp_state_dict)
            loc, scale, cov = physnp_batched(
                test_dataset,
                input_targets,
                model,
                gpu,
                nt,
                input_ts,
                n_samples=cfg.n_samples,
                mass_rhs_in=mass_rhs_in,
            )
            pred = rearrange(loc, "nf nt nx 1 -> nf (nx nt)")
            results["cov"] = cov
        elif isinstance(model, ANP) or isinstance(model, PINP) or isinstance(model, SoftcANP):
            loc, scale = anp_batched(
                test_dataset, input_targets, model, gpu, n_samples=cfg.n_samples
            )
            pred = rearrange(loc, "nf nx_nt 1 -> nf nx_nt")
        else:
            raise NotImplementedError()
        results.update({"loc": loc, "scale": scale, "pred": pred})
        torch.save(results, save_path)
    else:
        results = torch.load(save_path)
    return results


def truncate_results(results: Dict[str, Tensor]):
    # Enforce monotonicity
    loc = rearrange(results["loc"], "nf nt nx 1 -> nf nt nx")
    nf, nt, nx = loc.shape
    new_loc = loc.clone()
    for i in range(1, nx):
        new_loc[:, :, i] = new_loc[:, :, i] - F.relu(new_loc[:, :, i] - new_loc[:, :, i - 1])

    new_loc = F.relu(new_loc)

    pred = rearrange(new_loc, "nf nt nx -> nf (nx nt)")
    return {"loc": new_loc, "scale": results["scale"], "pred": pred}


def constrained_results(results: Dict[str, Tensor], mode: InequalityConstraint):
    # Enforce monotonicity
    new_loc = apply_non_linear_ineq_constraint(
        results["loc"], results["cov"], max_iter=10, mode=mode
    )
    pred = rearrange(new_loc, "nf nt nx 1 -> nf (nx nt)")
    return {"loc": new_loc, "scale": results["scale"], "pred": pred}


def anp_batched(test_dataset: ANPDataset, input_targets: Tensor, anp: ANP, gpu, n_samples: int):
    nf = test_dataset.tensors["input_contexts"].shape[0]
    n_functions_per_batch = 5
    anp = anp.to(gpu)
    it = input_targets.to(gpu).unsqueeze(0).expand(n_functions_per_batch, -1, -1)
    outputs = []
    with torch.no_grad():
        for i in range(0, nf, n_functions_per_batch):
            ic = test_dataset.tensors["input_contexts"][i : (i + n_functions_per_batch)]
            oc = test_dataset.tensors["output_contexts"][i : (i + n_functions_per_batch)]
            anp_dist = anp.get_loc_and_scale_batched(
                ic.to(gpu),
                oc.to(gpu),
                it,
                n_samples=n_samples,
                batch_size=50_000,
            )
            loc = anp_dist.loc.cpu()
            scale = anp_dist.scale.cpu()
            outputs.append((loc, scale))
        loc = torch.cat([x[0] for x in outputs])
        scale = torch.cat([x[1] for x in outputs])
    return loc, scale


def physnp_batched(
    test_dataset: ANPDataset,
    input_targets: Tensor,
    anp: PhysNP,
    gpu,
    nt: int,
    ts: int,
    n_samples: int,
    mass_rhs_in: Optional[Tensor] = None,
):
    nf = test_dataset.tensors["input_contexts"].shape[0]
    # nf = 5
    n_functions_per_batch = 5
    anp = anp.to(gpu)
    it = input_targets.to(gpu).unsqueeze(0).expand(n_functions_per_batch, -1, -1)
    it = rearrange(it, "nf (nx nt) d -> nf nt nx d", nt=nt)
    outputs = []
    ts = rearrange(ts, "nt -> 1 nt")
    with torch.no_grad():
        for i in range(0, nf, n_functions_per_batch):
            ic = test_dataset.tensors["input_contexts"][i : (i + n_functions_per_batch)]
            oc = test_dataset.tensors["output_contexts"][i : (i + n_functions_per_batch)]
            ic = rearrange(ic, "nf (nt nx) d -> nf nt nx d", nt=test_dataset.n_contexts_t)
            oc = rearrange(oc, "nf (nt nx) d -> nf nt nx d", nt=test_dataset.n_contexts_t)
            if mass_rhs_in is not None:
                mass_rhs = mass_rhs_in[i : (i + n_functions_per_batch)]
            else:
                mass_rhs = get_analytical_mass_rhs(test_dataset, i, i + n_functions_per_batch, ts)
            mass_rhs_i = mass_rhs.to(it.device)
            anp_dist, cov = anp.get_loc_and_scale_batched(
                ic.to(gpu),
                oc.to(gpu),
                it,
                n_samples=n_samples,
                batch_size=50_000,
                mass_rhs=mass_rhs_i.to(gpu),
            )
            loc = anp_dist.loc.cpu()
            scale = anp_dist.scale.cpu()
            outputs.append((loc, scale, cov))
        loc = torch.cat([x[0] for x in outputs])
        scale = torch.cat([x[1] for x in outputs])
        if outputs[0][2] is not None:
            cov = torch.cat([x[2] for x in outputs])
        else:
            cov = None
    return loc, scale, cov


def get_analytical_mass_rhs(test_dataset, i, i_end, ts):
    if isinstance(test_dataset, PorousMediumEquation):
        degree = test_dataset.degrees[i:i_end]
        degree = rearrange(degree, "nf -> nf 1")
        mass_rhs = mass_pme(degree, ts)
    elif isinstance(test_dataset, StefanPME):
        stefans = test_dataset.stefans[i:i_end]
        mass_rhs_list = []
        for stefan in stefans:
            mass_rhs_list.append(stefan.mass_stefan(ts))
        mass_rhs = torch.stack(mass_rhs_list, dim=0)
    elif isinstance(test_dataset, HeatEquation):
        nf = test_dataset.params.shape[0]
        nt = ts.shape[0]
        mass_rhs = torch.zeros((nf, nt), device=ts.device)
    elif isinstance(test_dataset, LinearAdvection):
        a_values = rearrange(test_dataset.parameters[i:i_end], "nf -> nf 1")
        mass_rhs = test_dataset.mass_advection(ts, a_values)
    elif isinstance(test_dataset, Burgers):
        a_values = rearrange(test_dataset.parameters[i:i_end], "nf -> nf 1")
        mass_rhs = mass_burgers(ts, a_values)
    else:
        raise NotImplementedError()
    return mass_rhs


def get_empirical_mass_rhs(cfg):
    dataset_path = Path(cfg.datasets.save_path)
    test_load_path = dataset_path / "test.pt"
    test_dataset = instantiate(cfg.datasets.test, load_path=test_load_path)
    _, output_targets, _ = get_test_solution(cfg, test_dataset)
    ot = rearrange(output_targets, "nf (nx nt) -> nf nx nt", nt=cfg.analysis.nt)
    return 0.5 * (
        reduce(ot[:, 1:], "nf nx nt -> nf nt", "mean")
        + reduce(ot[:, :-1], "nf nx nt -> nf nt", "mean")
    )


def get_test_solution(cfg, dataset: ANPDataset):
    nt: int = cfg.analysis.nt
    nx: int = cfg.analysis.nx

    tlims = dataset.lims("t")
    xlims = dataset.lims("x")

    ts = torch.linspace(*tlims, nt)
    xs = torch.linspace(*xlims, nx)
    inputs = meshgrid(ts.unsqueeze(0), xs.unsqueeze(0))
    # input_targets = rearrange(inputs, "nf nt nx d -> nf (nx nt) d")
    input_targets = rearrange(inputs, "nf nt nx d -> nf (nt nx) d")

    true_soln = dataset.solution(input_targets)
    output_targets = rearrange(true_soln, "nf (nt nx) 1 -> nf (nx nt)", nt=nt)

    input_targets = rearrange(input_targets, "1 (nt nx) d -> (nx nt) d", nt=nt)

    return input_targets, output_targets, ts


def estimate_shock_interval(
    mean: Tensor, cov: Tensor, n_samples=10, n_samples_per_batch: Optional[int] = None
):
    nf, nt, nx, _ = mean.shape
    outlist = []
    if n_samples_per_batch is None:
        n_samples_per_batch = n_samples
    for fid in tqdm(range(nf), desc="shock positions"):
        mean_i = mean[fid]
        cov_i = cov[fid]
        outlist_fid = []
        for _ in range(n_samples // n_samples_per_batch):
            first_less_than_zero = _estimate_shock_interval_for_one_f(
                mean_i, cov_i, n_samples_per_batch
            )
            outlist_fid.append(first_less_than_zero)
        outlist.append(torch.concat(outlist_fid, dim=0))
    return torch.stack(outlist, dim=1)


def _estimate_shock_interval_for_one_f(mean_i: Tensor, cov_i: Tensor, n_samples: int):
    nt, nx, _ = mean_i.shape
    device = mean_i.device
    idx = rearrange(torch.arange(0, nx, device=device), "nx -> 1 1 nx")
    try:
        chol: Tensor = torch.linalg.cholesky(cov_i).unsqueeze(0)
    except:
        chol: Tensor = torch.linalg.cholesky(
            cov_i + torch.eye(201).unsqueeze(0).cuda() * 1e-8
        ).unsqueeze(0)
    z = torch.randn(n_samples, *mean_i.shape, device=device)
    y = mean_i.unsqueeze(0) + chol.matmul(z)
    y = rearrange(y, "ns nt nx 1 -> ns nt nx")
    less_than_zero = y <= 0
    objective = less_than_zero * idx + (~less_than_zero) * nx
    return torch.argmin(objective, dim=2)


if __name__ == "__main__":
    main()
