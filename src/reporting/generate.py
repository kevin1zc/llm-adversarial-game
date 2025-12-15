from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in values:
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _write_wide_csv(path: Path, columns: Dict[str, List[float]]) -> None:
    keys = sorted(columns.keys())
    n = max((len(columns[k]) for k in keys), default=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            row = [columns[k][i] if i < len(columns[k]) else "" for k in keys]
            w.writerow(row)


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_training_figures(
    out_dir: Path,
    train: Dict[str, List[float]],
    eval_: Dict[str, List[float]],
    window: int = 50,
    update_mode: Optional[str] = None,
    max_turns: Optional[int] = None,
) -> List[Path]:
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    figs: List[Path] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_rolling_mean_w{int(window)}" if int(window) > 1 else ""

    def _save(fig, name: str, add_suffix: bool) -> None:
        sfx = suffix if add_suffix else ""
        p_png = fig_dir / f"{name}{sfx}.png"
        fig.savefig(p_png, dpi=200, bbox_inches="tight")
        figs.append(p_png)
        plt.close(fig)

    def _rolling_mean_masked(values: List[float], window: int, invalid: float = -1.0) -> List[float]:
        out: List[float] = []
        buf: List[float] = []
        for v in values:
            if v >= invalid + 1e-9:
                buf.append(float(v))
            if len(buf) > window:
                buf = buf[-window:]
            out.append(sum(buf) / len(buf) if buf else float("nan"))
        return out

    episodes = list(range(1, len(train.get("defender_reward", [])) + 1))
    if episodes:
        window = max(1, min(int(window), max(5, len(episodes) // 4)))
    r_def = _rolling_mean(train.get("defender_reward", []), window)
    r_adv = _rolling_mean(train.get("adversary_reward", []), window)
    if update_mode == "alternating":
        def_losses = []
        adv_losses = []
        for i, v in enumerate(train.get("defender_loss", [])):
            def_losses.append(float(v) if (i % 2 == 0) else float("nan"))
        for i, v in enumerate(train.get("adversary_loss", [])):
            adv_losses.append(float(v) if (i % 2 == 1) else float("nan"))
        r_loss_def = _rolling_mean_masked([(-1.0 if (x != x) else x) for x in def_losses], window)
        r_loss_adv = _rolling_mean_masked([(-1.0 if (x != x) else x) for x in adv_losses], window)
    else:
        r_loss_def = _rolling_mean(train.get("defender_loss", []), window)
        r_loss_adv = _rolling_mean(train.get("adversary_loss", []), window)

    if "train_asr" in train:
        r_asr = _rolling_mean_masked(train.get("train_asr", []), window)
        r_over_ref = _rolling_mean_masked(train.get("train_over_refusal", []), window)
        r_ben_util = _rolling_mean_masked(train.get("train_benign_utility", []), window)
    else:
        r_asr = _rolling_mean(train.get("violation_rate", []), window)
        r_over_ref = _rolling_mean(train.get("refusal_rate", []), window)
        r_ben_util = _rolling_mean(train.get("defender_reward", []), window)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax1.plot(episodes, r_def, label="defender_reward")
    ax1.plot(episodes, r_adv, alpha=0.6, label="adversary_reward")
    ax1.set_title("Train rewards")
    ax1.set_xlabel("episode")
    ax1.legend(fontsize=8)

    ax2.plot(episodes, r_loss_def, label="defender_loss (updated episodes)")
    ax2.plot(episodes, r_loss_adv, alpha=0.6, label="adversary_loss (updated episodes)")
    ax2.set_title("Policy gradient losses")
    ax2.set_xlabel("episode")
    ax2.legend(fontsize=8)

    ax3.plot(episodes, r_asr, label="train_asr (malicious only)")
    ax3.set_title("Train ASR (malicious only)")
    ax3.set_xlabel("episode")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=8)

    ax4.set_title("Benign-only metrics")
    ax4.set_xlabel("episode")
    ax4.set_ylim(0.0, 1.0)
    l1 = ax4.plot(episodes, r_over_ref, label="train_over_refusal (benign only)")
    ax4.set_ylabel("over-refusal rate")

    ax4b = ax4.twinx()
    l2 = ax4b.plot(episodes, r_ben_util, label="train_benign_utility (per-turn, benign only)", color="C1", alpha=0.7)
    ax4b.set_ylabel("benign utility")

    lines = (l1 + l2)
    labels = [ln.get_label() for ln in lines]
    ax4.legend(lines, labels, fontsize=8, loc="best")

    _save(fig, "train_curves", add_suffix=True)

    if eval_.get("eval_episode"):
        x = eval_["eval_episode"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax1.plot(x, eval_.get("eval_attack_success_rate", []), marker="o")
        ax1.set_title("ASR")
        ax1.set_xlabel("episode")
        ax1.set_ylim(0.0, 1.0)

        ax2.plot(x, eval_.get("eval_over_refusal_rate", []), marker="o")
        ax2.set_title("over_refusal_rate")
        ax2.set_xlabel("episode")
        ax2.set_ylim(0.0, 1.0)

        ax3.plot(x, eval_.get("eval_benign_utility", []), marker="o")
        ax3.set_title("benign_utility")
        ax3.set_xlabel("episode")

        vt = [float(v) for v in eval_.get("eval_avg_violation_turn", [])]
        if max_turns is not None:
            cap = float(int(max_turns) + 1)
            vt = [cap if (v == float("inf") or v != v) else v for v in vt]
        ax4.plot(x, vt, marker="o")
        ax4.set_title("avg_violation_turn")
        ax4.set_xlabel("episode")

        _save(fig, "eval_curves", add_suffix=False)

    return figs


def generate_run_report(
    experiment_dir: Path,
    metrics: Dict[str, Any],
    hparams: Optional[Dict[str, Any]] = None,
    smoothing_window: int = 50,
) -> Dict[str, Any]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = experiment_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_keys = [
        "defender_reward",
        "adversary_reward",
        "episode_length",
        "defender_loss",
        "adversary_loss",
        "is_benign",
        "train_asr",
        "train_over_refusal",
        "train_benign_utility",
        "train_violation_turn",
    ]
    eval_keys = [
        "eval_episode",
        "eval_attack_success_rate",
        "eval_over_refusal_rate",
        "eval_benign_utility",
        "eval_avg_violation_turn",
    ]

    train = {k: [float(x) for x in metrics.get(k, [])] for k in train_keys}
    eval_ = {k: [float(x) for x in metrics.get(k, [])] for k in eval_keys}

    _write_wide_csv(tables_dir / "metrics_train.csv", train)
    _write_wide_csv(tables_dir / "metrics_eval.csv", eval_)

    summary_rows: List[Dict[str, Any]] = []
    n = len(train.get("defender_reward", []))
    last_k = min(100, n) if n > 0 else 0
    if last_k > 0:
        def_last = train["defender_reward"][-last_k:]
        adv_last = train["adversary_reward"][-last_k:]
        asr_last = [x for x in train.get("train_asr", [])[-last_k:] if x >= 0.0]
        over_last = [x for x in train.get("train_over_refusal", [])[-last_k:] if x >= 0.0]
        util_last = [x for x in train.get("train_benign_utility", [])[-last_k:] if x >= 0.0]
        vt_last = [x for x in train.get("train_violation_turn", [])[-last_k:] if x >= 0.0]
        summary_rows.append(
            {
                "split": "train_last100",
                "defender_reward_mean": sum(def_last) / len(def_last),
                "adversary_reward_mean": sum(adv_last) / len(adv_last),
                "attack_success_rate": (sum(asr_last) / len(asr_last)) if asr_last else "",
                "over_refusal_rate": (sum(over_last) / len(over_last)) if over_last else "",
                "benign_utility": (sum(util_last) / len(util_last)) if util_last else "",
                "avg_violation_turn": (sum(vt_last) / len(vt_last)) if vt_last else "",
            }
        )

    if eval_.get("eval_episode"):
        i_last = len(eval_["eval_episode"]) - 1
        summary_rows.append(
            {
                "split": "eval_last",
                "episode": eval_["eval_episode"][i_last],
                "attack_success_rate": eval_.get("eval_attack_success_rate", [""])[i_last]
                if i_last < len(eval_.get("eval_attack_success_rate", []))
                else "",
                "over_refusal_rate": eval_.get("eval_over_refusal_rate", [""])[i_last]
                if i_last < len(eval_.get("eval_over_refusal_rate", []))
                else "",
                "benign_utility": eval_.get("eval_benign_utility", [""])[i_last]
                if i_last < len(eval_.get("eval_benign_utility", []))
                else "",
                "avg_violation_turn": eval_.get("eval_avg_violation_turn", [""])[i_last]
                if i_last < len(eval_.get("eval_avg_violation_turn", []))
                else "",
            }
        )

    _write_summary_csv(tables_dir / "summary.csv", summary_rows)
    with (tables_dir / "summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)

    if hparams is not None:
        with (tables_dir / "hparams.json").open("w") as f:
            json.dump(hparams, f, indent=2)

    figs = _plot_training_figures(
        experiment_dir,
        train,
        eval_,
        window=smoothing_window,
        update_mode=(hparams or {}).get("update_mode"),
        max_turns=(hparams or {}).get("max_turns"),
    )
    return {
        "experiment_dir": str(experiment_dir),
        "tables": [str(tables_dir / "metrics_train.csv"), str(tables_dir / "metrics_eval.csv"), str(tables_dir / "summary.csv")],
        "figures": [str(p) for p in figs],
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=str, required=True)
    p.add_argument("--metrics-json", type=str, required=True)
    p.add_argument("--hparams-json", type=str, default=None)
    p.add_argument("--smoothing-window", type=int, default=50)
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir)
    with Path(args.metrics_json).open("r") as f:
        metrics = json.load(f)
    hparams = None
    if args.hparams_json:
        with Path(args.hparams_json).open("r") as f:
            hparams = json.load(f)

    generate_run_report(
        experiment_dir=experiment_dir,
        metrics=metrics,
        hparams=hparams,
        smoothing_window=int(args.smoothing_window),
    )


if __name__ == "__main__":
    main()


