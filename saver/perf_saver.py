# Checkpointing, per-term logging, and the per-eval-step curve plot.
#
# The log records EVERY loss term separately, not just the total. With four terms whose
# weights are the thing under study, a single scalar makes the sweep unreadable: a run whose
# total went down because the FM term collapsed while L1 got worse looks identical to a run
# that genuinely improved. Every term is written unweighted (its raw value) AND weighted (its
# actual contribution), so both "did this term improve" and "did this term matter" are
# answerable from the log alone.
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class Saver:
    def __init__(self, out_dir: Path, keep_last_n: int = 3):
        self.out = Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "plots").mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "log.jsonl"
        self.keep_last_n = keep_last_n
        self.best = float("inf")

    def log(self, record: Dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_config(self, cfg: Dict) -> None:
        with open(self.out / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def save_checkpoint(self, model, opt, epoch: int, val: float, tag: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> None:
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "epoch": epoch, "val": val, **(metadata or {})}
        torch.save(ck, self.out / "checkpoints" / "last.pt")
        if tag:
            torch.save(ck, self.out / "checkpoints" / f"{tag}.pt")
        # Kept as a health signal only. Selection for this experiment is by the audio-level
        # eval, because a lower total loss can just mean a differently-weighted mix of terms.
        if val < self.best:
            self.best = val
            torch.save(ck, self.out / "checkpoints" / "best.pt")

    # ---- plots ------------------------------------------------------------------------
    @staticmethod
    def _spans(flag: np.ndarray):
        """Contiguous [start, end) runs of a boolean frame flag."""
        f = np.asarray(flag).astype(bool)
        if not f.any():
            return []
        d = np.diff(f.astype(np.int8), prepend=0, append=0)
        return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

    def plot_curves(self, epoch: int, recording: str, pred: np.ndarray, target: np.ndarray,
                    valid: np.ndarray, n_strings: int, extra: Optional[Dict] = None,
                    active: Optional[np.ndarray] = None,
                    onset: Optional[np.ndarray] = None,
                    pitch: Optional[tuple] = None) -> Path:
        """Predicted vs real envelope for every string in one val recording. Written at every
        eval step, not only at the end -- a run that collapses to a flat line is obvious here
        an hour in, and invisible in the loss until much later.

        `active` shades the frames a note is sounding on that string (the score-derived
        voiced flag). Without it a flat stretch is ambiguous: it reads the same whether the
        model failed to produce a decay or whether nothing was being played there.

        `onset` draws each excitation instant. These are the REALIGNED onsets
        (dataset/control/onset_align.py), a median 27 ms earlier than GuitarSet's annotation,
        so an attack should line up with its own line -- if predicted attacks sit
        consistently to one side, that is a segmentation problem and not a model problem.
        """
        T = pred.shape[0] // n_strings
        p = pred.reshape(n_strings, T)
        t = target.reshape(n_strings, T)
        v = valid.reshape(n_strings, T)
        a = active.reshape(n_strings, T) if active is not None else None
        o = onset.reshape(n_strings, T) if onset is not None else None
        # optional second row per string: predicted vs real PITCH, on its own axis (pitch and
        # log-RMS share no scale, so overlaying them would make one of the two unreadable)
        pp = (pitch[0].reshape(n_strings, T), pitch[1].reshape(n_strings, T)) if pitch else None
        rows_per_string = 2 if pp else 1

        fig, axes = plt.subplots(n_strings * rows_per_string, 1,
                                 figsize=(14, 1.9 * n_strings * rows_per_string),
                                 sharex=True, squeeze=False)
        for s in range(n_strings):
            ax = axes[s * rows_per_string][0]
            x = np.arange(T)
            m = v[s]
            if a is not None:
                for lo, hi in self._spans(a[s] > 0.5):
                    ax.axvspan(lo, hi, color="0.55", alpha=0.22, lw=0)
            if o is not None:
                # the impulse channel decays by 0.6 per frame, so only the onset frame is 1.0
                for f in np.where(o[s] >= 0.999)[0]:
                    ax.axvline(f, color="tab:blue", lw=0.8, alpha=0.7)
            ax.plot(x[m], t[s][m], color="black", lw=1.2, label="real")
            ax.plot(x[m], p[s][m], color="tab:red", lw=1.1, ls="--", label="predicted")
            ax.set_ylabel(f"s{s+1} env", fontsize=9)
            if pp is not None:
                axp = axes[s * rows_per_string + 1][0]
                for lo, hi in (self._spans(a[s] > 0.5) if a is not None else []):
                    axp.axvspan(lo, hi, color="0.55", alpha=0.22, lw=0)
                if o is not None:
                    for f in np.where(o[s] >= 0.999)[0]:
                        axp.axvline(f, color="tab:blue", lw=0.8, alpha=0.7)
                axp.plot(x[m], pp[1][s][m], color="black", lw=1.2)
                axp.plot(x[m], pp[0][s][m], color="tab:green", lw=1.1, ls="--")
                axp.set_ylabel(f"s{s+1} pitch", fontsize=9)
            if s == 0:
                if a is not None:
                    ax.axvspan(0, 0, color="0.55", alpha=0.22, lw=0, label="note active")
                if o is not None:
                    ax.plot([], [], color="tab:blue", lw=0.8, alpha=0.7, label="onset")
                ax.legend(fontsize=8, loc="upper right")
        axes[-1][0].set_xlabel("frame")
        title = f"{recording} — envelope, epoch {epoch}"
        if extra:
            title += "  |  " + "  ".join(f"{k}={v:.3f}" for k, v in extra.items())
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        path = self.out / "plots" / f"curves_ep{epoch:04d}.png"
        fig.savefig(path, dpi=120); plt.close(fig)
        return path

    def plot_loss_terms(self) -> Optional[Path]:
        """Every logged term on its own axis, train and val together. One panel per term is
        deliberate -- terms differ by orders of magnitude and sharing an axis hides all but
        the largest."""
        if not self.log_path.exists():
            return None
        rows = [json.loads(l) for l in open(self.log_path)]
        rows = [r for r in rows if r.get("split") == "train"]
        if len(rows) < 2:
            return None
        val_rows = [json.loads(l) for l in open(self.log_path) if json.loads(l).get("split") == "val"]

        terms = [k for k in rows[-1] if k.startswith("loss_") or k in ("loss", "roughness_ratio",
                                                                      "variance_ratio")]
        if not terms:
            return None
        n = len(terms)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 2.4 * ((n + 1) // 2)), squeeze=False)
        for i, term in enumerate(terms):
            ax = axes[i // 2][i % 2]
            xs = [r["epoch"] for r in rows if term in r]
            ys = [r[term] for r in rows if term in r]
            ax.plot(xs, ys, color="tab:blue", lw=1.2, label="train")
            vx = [r["epoch"] for r in val_rows if term in r]
            vy = [r[term] for r in val_rows if term in r]
            if vx:
                ax.plot(vx, vy, color="tab:orange", lw=1.2, label="val")
            ax.set_title(term, fontsize=9)
            ax.set_xlabel("epoch")
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(n, ((n + 1) // 2) * 2):
            axes[j // 2][j % 2].axis("off")
        fig.tight_layout()
        path = self.out / "plots" / "loss_terms.png"
        fig.savefig(path, dpi=120); plt.close(fig)
        return path
