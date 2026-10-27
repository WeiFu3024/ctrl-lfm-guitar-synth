from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:  # annotation only -- model/common/ must not import model/synth/ at runtime,
    # and the objective is decoder-agnostic in any case: it is used with FMDecoder and with
    # JointSynthDecoder, which only ever agree on the rank-3 (B, L, D) contract.
    from model.synth.fm_decoder import FMDecoder


class FlowMatching:
    """Rectified-flow training objective and Euler sampler, over any rank-3 (B, L, D)
    decoder -- FMDecoder per string, JointSynthDecoder over all six at once."""

    def __init__(self, model: "FMDecoder", cfg):
        self.model = model
        self.cfg = cfg

    def _sample_t(self, batch_size: int, device: torch.device) -> Tensor:
        if self.cfg.t_distribution == "logit_normal":
            z = torch.randn(batch_size, device=device) * self.cfg.logit_normal_std + self.cfg.logit_normal_mean
            return torch.sigmoid(z)
        if self.cfg.t_distribution == "uniform":
            return torch.rand(batch_size, device=device)
        raise ValueError(f"Unknown t_distribution: {self.cfg.t_distribution}")

    def compute_loss(
        self,
        x1: Tensor,
        controls: Dict[str, Tensor],
        valid_mask: Optional[Tensor] = None,
        cfg_dropout_prob: Optional[float] = None,
        return_x1_hat: bool = False,
    ):
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        if valid_mask is not None:
            # Random padded tokens otherwise participate in temporal attention and perturb
            # valid frames even though their own loss is masked. Keep padding deterministic,
            # matching the zero-padded controls used by every variable-length collator.
            data_mask = valid_mask.unsqueeze(-1).to(x1.dtype)
            x0 = x0 * data_mask
            x1 = x1 * data_mask
        t = self._sample_t(B, x1.device)
        t_ = t.view(B, 1, 1)
        x_t = (1 - t_) * x0 + t_ * x1
        v_target = x1 - x0

        # None -> use the configured training-time default (self.cfg.cfg_dropout_prob);
        # callers (e.g. validation) can pass 0.0 to always evaluate the conditional path
        dropout_prob = self.cfg.cfg_dropout_prob if cfg_dropout_prob is None else cfg_dropout_prob
        drop_mask = torch.rand(B, device=x1.device) < dropout_prob
        v_pred = self.model(x_t, t, controls, cfg_drop_mask=drop_mask)

        sq_err = (v_pred - v_target) ** 2  # (B,T,D)
        if valid_mask is None:
            loss = sq_err.mean()
        else:
            mask = valid_mask.unsqueeze(-1).to(sq_err.dtype)  # (B,T,1)
            denom = (mask.sum() * sq_err.shape[-1]).clamp(min=1)
            loss = (sq_err * mask).sum() / denom

        if not return_x1_hat:
            return loss
        # Standard flow-matching/diffusion trick: a one-step estimate of the clean
        # sample from the current noisy point + predicted velocity, algebraically
        # derived from x_t=(1-t)x0+t*x1, v=x1-x0 -- no extra model forward pass needed.
        # Used by the GAN path (trainer/note_score_trainer.py) as a cheap differentiable
        # proxy for "what the model currently thinks x1 looks like", avoiding having to
        # unroll the full N-step sampler during training.
        x1_hat = x_t + (1 - t_) * v_pred
        return loss, x1_hat

    @torch.no_grad()
    def sample(
        self,
        controls: Dict[str, Tensor],
        shape: Tuple[int, int, int],
        n_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        device: str = "cuda",
        x0: Optional[Tensor] = None,
    ) -> Tensor:
        n_steps = n_steps or self.cfg.sampling_steps
        cfg_scale = self.cfg.default_cfg_scale if cfg_scale is None else cfg_scale

        B, _, _ = shape
        # x0 lets a caller supply the starting noise instead of drawing it here. Needed for
        # rendering a recording longer than the trained window in OVERLAPPING slices: giving
        # adjacent slices the SAME noise on the frames they share makes their outputs agree
        # there, so the overlap can be blended. With independently drawn noise the two slices
        # are different samples of the same conditional distribution and blending them
        # averages two unrelated realisations.
        x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
        ts = torch.linspace(0, 1, n_steps + 1, device=device)

        for i in range(n_steps):
            t_cur = ts[i].expand(B)
            dt = ts[i + 1] - ts[i]

            if cfg_scale != 1.0:
                v_cond = self.model(x, t_cur, controls, cfg_drop_mask=torch.zeros(B, dtype=torch.bool, device=device))
                v_uncond = self.model(x, t_cur, controls, cfg_drop_mask=torch.ones(B, dtype=torch.bool, device=device))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(x, t_cur, controls, cfg_drop_mask=None)

            x = x + dt * v

        return x
