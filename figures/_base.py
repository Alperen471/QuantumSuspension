"""figures/_base.py — Ortak stil, renkler, yardımcı fonksiyonlar."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

# ── Stil ──────────────────────────────────────────────────────────── #

def set_style():
    plt.rcParams.update({
        "font.family"       : "serif",
        "font.size"         : 10,
        "axes.titlesize"    : 11,
        "axes.labelsize"    : 10,
        "xtick.labelsize"   : 9,
        "ytick.labelsize"   : 9,
        "legend.fontsize"   : 9,
        "figure.dpi"        : 300,
        "savefig.dpi"       : 300,
        "savefig.bbox"      : "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid"         : True,
        "grid.alpha"        : 0.3,
        "lines.linewidth"   : 1.5,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })

# ── Renkler / etiketler ───────────────────────────────────────────── #

MODEL_COLORS = {
    "passive"             : "#888888",
    "skyhook"             : "#444444",
    "sac_baseline"        : "#4878CF",
    "td3_baseline"        : "#E45B5B",
    "sac_fuzzy"           : "#6ACC65",
    "td3_fuzzy"           : "#F7910A",
    "sac_q5_qfm"          : "#B47CC7",
    "sac_q5_qfm_fuzzy"    : "#C4AD66",
    "sac_q10_qfm"         : "#77BEDB",
    "sac_q10_qfm_fuzzy"   : "#56A64B",
    "sac_q20_qfm"         : "#D65F5F",
    "sac_q20_qfm_fuzzy"   : "#9966AA",
    # ablation dosya adlarından gelen isimlere de eşle
    "rl_baseline"         : "#4878CF",
    "rl_fuzzy_bounded"    : "#6ACC65",
    "rl_q5_var"           : "#B47CC7",
    "rl_q5_var_fuzzy"     : "#C4AD66",
}

MODEL_LABELS = {
    "passive"             : "Passive",
    "skyhook"             : "Skyhook",
    "sac_baseline"        : "SAC Baseline",
    "td3_baseline"        : "TD3 Baseline",
    "sac_fuzzy"           : "SAC + Fuzzy",
    "td3_fuzzy"           : "TD3 + Fuzzy",
    "sac_q5_qfm"          : "SAC + QFM-5q",
    "sac_q5_qfm_fuzzy"    : "SAC + QFM-5q + Fuzzy",
    "sac_q10_qfm"         : "SAC + QFM-10q",
    "sac_q10_qfm_fuzzy"   : "SAC + QFM-10q + Fuzzy",
    "sac_q20_qfm"         : "SAC + QFM-20q",
    "sac_q20_qfm_fuzzy"   : "SAC + QFM-20q + Fuzzy",
    "rl_baseline"         : "SAC Baseline",
    "rl_fuzzy_bounded"    : "SAC + Fuzzy",
    "rl_q5_var"           : "SAC + QFM-5q",
    "rl_q5_var_fuzzy"     : "SAC + QFM-5q + Fuzzy",
}

# ── Kayıt ─────────────────────────────────────────────────────────── #

def save(path: str, fig=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(str(p))
    plt.close("all")
    print(f"  ✓ {p}")

# ── Sonuç yükleme ─────────────────────────────────────────────────── #

def load_ablation(results_dir: str) -> dict:
    """JSON ablation sonuçlarını yükle."""
    d = Path(results_dir)
    sf = d / "ablation_summary.json"
    if sf.exists():
        with open(sf) as f:
            raw = json.load(f)
        return {m: _summarize(runs) for m, runs in raw.items() if runs}

    grouped = {}
    for f in d.glob("*.json"):
        try:
            with open(f) as fp:
                run = json.load(fp)
            m = run.get("model_name", "?")
            grouped.setdefault(m, []).append(run)
        except Exception:
            pass
    return {m: _summarize(runs) for m, runs in grouped.items()}


def _summarize(runs):
    keys = ["rms_body_acc_mean", "iso_weighted_rms_mean",
            "rms_ctrl_force_mean", "rms_susp_defl_mean"]
    out  = {"n_runs": len(runs), "eval_history": []}
    for k in keys:
        vals = [r.get("final_metrics", {}).get(k)
                for r in runs if r.get("final_metrics", {}).get(k) is not None]
        if vals:
            out[f"{k}_avg"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    by_road = {}
    for run in runs:
        rt  = run.get("road_type", "?")
        val = run.get("final_metrics", {}).get("rms_body_acc_mean")
        if val is not None:
            by_road.setdefault(rt, []).append(val)
    out["by_road"] = {
        rt: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for rt, v in by_road.items()
    }
    # eval history varsa
    for run in runs:
        if run.get("eval_metrics"):
            out["eval_history"] = run["eval_metrics"]
            break
    return out


def fmt_steps(ax):
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}K"))
