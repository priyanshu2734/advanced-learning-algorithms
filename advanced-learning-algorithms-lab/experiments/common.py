"""Shared plotting helpers."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def out_path(name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / name


def plot_history(history, title, path, keys=(("loss", "val_loss"), ("error", "val_error"))):
    fig, axes = plt.subplots(1, len(keys), figsize=(10, 3.6))
    for ax, (tr, va) in zip(axes, keys):
        ax.plot(history[tr], label="train")
        if va in history:
            ax.plot(history[va], label="cross-validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(tr)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
