"""Figures + REPORT.md for Detox loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from detox.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(methods, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(methods):
        hist = m["history"]
        ax.plot(
            [h["epoch"] for h in hist],
            [h["loss"] for h in hist],
            color=MULTI_SERIES_COLORS[i % len(MULTI_SERIES_COLORS)],
            linewidth=LINEWIDTH_MAIN,
            label=m["method"],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Softmax vs Detox (UTKFace gender)")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_fairness(methods: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = list(methods.keys())
    x = range(len(names))
    w = 0.25
    bacc = [methods[n]["balanced_accuracy"] * 100 for n in names]
    ea = [methods[n].get("bacc_ea", 0) * 100 for n in names]
    eb = [methods[n].get("bacc_eb", 0) * 100 for n in names]
    ax.bar([i - w for i in x], bacc, w, label="Overall BAcc", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.bar(list(x), ea, w, label="E-A BAcc", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.bar([i + w for i in x], eb, w, label="E-B BAcc", color=MULTI_SERIES_COLORS[2], edgecolor="#002080")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("%")
    ax.set_title("Balanced accuracy by ethnicity")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    methods = eval_res.get("methods", {})
    lines = [
        "# REPORT — Detox Loss: Fairness Constraints for Imbalanced Data",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        f"Lab twin: gender classification on UTKFace Protocol-{train_s.get('utk_protocol')} "
        f"(E-B ratio={train_s.get('eb_ratio')}); Softmax vs Detox (Exclusion+Inclusion+F-Distill).",
        "",
        "## Training",
        "",
        f"- Task: {train_s.get('task')}",
        f"- Train images: {train_s.get('n_train')} ({train_s.get('race_counts')})",
        f"- Methods: {', '.join(m['method'] for m in train_s.get('methods', []))}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Fairness evaluation (balanced test)",
        "",
        "| Method | BAcc | E-A BAcc | E-B BAcc | |BAcc gap| |",
        "|--------|------|----------|----------|-----------|",
    ]
    for name, v in methods.items():
        gap = v.get("bacc_gap")
        gaps = f"{gap*100:.2f}%" if gap is not None else "—"
        lines.append(
            f"| {name} | {v['balanced_accuracy']*100:.2f}% | "
            f"{v.get('bacc_ea', 0)*100:.2f}% | {v.get('bacc_eb', 0)*100:.2f}% | {gaps} |"
        )
    lines += [
        "",
        f"![Fairness]({fig_dir.as_posix()}/fairness_metrics.svg)",
        "",
        "## Notes",
        "",
        "- Paper also reports age-group + PPB; this lab twin focuses on gender under Protocol-1 with E-B=10%.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["methods"], fig_dir / "train_loss.svg")
    plot_fairness(eval_res["methods"], fig_dir / "fairness_metrics.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
