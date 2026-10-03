# The 6->1 mixing stage: one learned impulse response per string, convolved and summed
# into a mono signal that should match the room-mic recording.
#
# Ported from reference/ddsp-guitar/synth.py's MultiChannelReverb so the comparison
# against DDSP-Guitar uses the same mixing component they do -- their checkpoint carries
# impulse_response_param of shape (6, 12000), i.e. 0.25 s per string at their 48 kHz.
# Their weights are NOT reused: they were fit at 48 kHz on their synth's own unsupervised
# per-voice signals, whereas ours must map SAO-decoded hex-pickup audio at 44.1 kHz.
#
# Why one linear filter is the right model here: pickup -> mic is dominated by body
# radiation and room response, both approximately linear and time-invariant. A causal FIR
# also absorbs the mic's propagation delay (~1 ms at typical distance) and any level
# difference for free, so no alignment or gain-matching pass is needed -- the learned IR
# simply places its peak at the true delay.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STRINGS = 6
IR_DURATION_S = 0.25  # their ir_duration; 11025 taps at 44.1 kHz


class MultiChannelReverb(nn.Module):
    """(B, S, N) per-string audio -> (B, N) mono mix."""

    def __init__(self, sample_rate: int, ir_duration: float = IR_DURATION_S,
                 n_channels: int = N_STRINGS, init_scale: float = 0.01,
                 residual: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        # residual=True reproduces DDSP-Guitar's topology: synthesis_model.py:143 returns
        # dry_mix + reverb(dry), so the EFFECTIVE ir per string is (delta + h_s) -- the dry
        # path carries the direct sound and the learned filter only adds reflections. That is
        # why get_ir() zeroes tap 0; we had copied the zeroing without the dry path.
        #
        # TESTED AND IT DOES NOT HELP -- kept as a switch, off by default, so the negative
        # result stays reproducible. Same per-string audio, only the reverb swapped, 36 test
        # recordings, mix level vs the real mic:
        #
        #     baseline  sd 4.64 dB   corr(room gain) -0.928   val_loss 7.0850
        #     residual  sd 4.56 dB   corr(room gain) -0.933   val_loss 7.0903
        #     DDSP-Guitar's own      sd 3.09 dB      corr -0.528
        #
        # The hypothesis was that a dry path anchors the overall gain, which is why DDSP
        # scatters less. It is wrong: h_s is free, so the optimiser simply learns
        # h_s - delta and lands on the same function. A skip connection reparameterises the
        # model without changing its function space, so under the same loss it converges to
        # the same solution. Anchoring the gain needs a constraint on h_s (normalise IR
        # energy, or split gain from filter shape), not a skip.
        #
        # The remaining suspect for DDSP's lower scatter is TRAINING CONDITIONS, not
        # topology, and their code says so directly: train_synthesis.py:146 takes the
        # multiscale spectral loss against inputs["mic_audio"], so the MICROPHONE is the
        # training target of their whole synth. Absolute mic level is what they optimise.
        # Ours is trained against per-string hex latents, and mic level is seen only by this
        # module, fitted afterwards on frozen SAO-decoded audio that cannot move. Measured on
        # the 36 test recordings (scripts/pipeline/plot_level_accuracy.py), level vs the real
        # mic:
        #
        #     arm         bias      sd    MAE      r   slope
        #     hex sum    +8.66    4.79   8.74  0.336    0.20    <- the input, i.e. the job
        #     ours dry   +8.18    4.82   8.31  0.314    0.17
        #     ours wet   -2.22    4.51   3.92  0.477    0.35
        #     ddsp       -0.93    3.15   2.61  0.769    0.56
        #
        # So this module removes almost all of the +8.7 dB offset (MAE 8.3 -> 3.9) but barely
        # improves TRACKING (slope 0.17 -> 0.35): it learns the average room, not this room.
        #
        # What no topology fixes: one fixed IR against GuitarSet's hex->mic gain, which spans
        # -26.7 .. +4.5 dB over all 360 recordings, sd 5.92 dB
        # (scripts/data/analyze_room_gain.py). That spread is NOT six per-player rooms -- it
        # is TWO clusters, players 00,01 at a median -13.7 dB and 02-05 at -5.2 dB. Player
        # explains eta^2 = 0.451 of the variance, a binary split of exactly those two groups
        # explains 0.445 of it, and within either cluster player explains nothing (p = 0.34
        # and 0.62). Genre, style, tempo and key explain none of it (eta^2 <= 0.03, p > 0.3),
        # so it is a recording-setup variable, not a musical one. Conditioning on it would
        # cut the residual sd from 5.92 to 4.39 dB -- real, but it leaves more than half the
        # variance, so it is a partial fix rather than the answer.
        self.residual = residual
        self.ir_samples = int(ir_duration * sample_rate)
        # same small-random init as theirs; a zero init leaves the output silent and the
        # spectral loss' log term undefined on the first step
        self.impulse_response_param = nn.Parameter(
            torch.rand((n_channels, self.ir_samples), dtype=torch.float32) * init_scale)

    def get_ir(self) -> torch.Tensor:
        # First tap forced to zero, as reference/ddsp-guitar/synth.py does. In THEIR design
        # this pairs with the dry path (see residual in __init__): the delta supplies the
        # direct sound, so tap 0 of the learned filter must be empty. With residual=False the
        # zeroing is inherited without its reason -- kept for backward compatibility with the
        # checkpoints already trained this way.
        zeros = torch.zeros((self.n_channels, 1), dtype=self.impulse_response_param.dtype,
                            device=self.impulse_response_param.device)
        return torch.cat((zeros, self.impulse_response_param[:, 1:]), dim=-1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """signal (B, S, N) -> (B, N). Filters each string with its own IR and sums them --
        the mixing IS the convolution.

        Done in the frequency domain: a direct 11025-tap conv1d over 8 s of 6-channel audio
        is ~93 GFLOPs per batch of 4, and a 120-epoch fit had not finished one scheme after
        an hour -- absurd for a 66k-parameter model. FFT makes it O(N log N): measured
        25x faster (2116 -> 85 ms fwd+bwd per batch of 4) and agreeing with
        F.conv1d(..., padding="same") to 4e-4 max relative error, which is fp32 FFT
        accumulation over ~350k samples, not a logic difference.
        """
        ir = self.get_ir()                                        # (S, L)
        L, N = ir.shape[-1], signal.shape[-1]
        n_fft = 1 << (N + L - 1).bit_length()
        # conv1d computes cross-correlation, so flip the kernel to express it as convolution
        Xf = torch.fft.rfft(signal, n=n_fft)                      # (B, S, F)
        Wf = torch.fft.rfft(torch.flip(ir, dims=(-1,)), n=n_fft)  # (S, F)
        full = torch.fft.irfft((Xf * Wf[None]).sum(dim=1), n=n_fft)
        off = (L - 1) - (L - 1) // 2                              # matches padding="same"
        wet = full[..., off:off + N]
        return wet + signal.sum(dim=1) if self.residual else wet

    def per_string(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, S, N) -> (B, S, N), each string filtered but NOT summed. For inspecting what
        the model decided each string contributes."""
        ir = self.get_ir()
        out = torch.zeros_like(signal)
        for i in range(self.n_channels):
            out[:, i] = F.conv1d(signal[:, i:i + 1], ir[i][None, None, :], padding="same").squeeze(1)
        return out


def multiscale_spectral_loss(a: torch.Tensor, b: torch.Tensor, scales, overlap: float = 0.75,
                             lin_weight: float = 1.0, log_weight: float = 1.0) -> torch.Tensor:
    """reference/ddsp-guitar/loss.py's multiscale_spectral_loss (linear + log magnitude).

    Phase-invariant, which matters because the reverb is also trained on *generated* audio
    that is perceptually right but not phase-aligned with the recording. On oracle inputs
    the problem is actually convex in the time domain (see --time-weight in
    scripts/synth/train_mix_reverb.py), but only the spectral term transfers to all variants.
    """
    loss = torch.zeros((), device=a.device)
    for s in scales:
        kw = dict(n_fft=s, hop_length=int(s * (1 - overlap)), win_length=s,
                  window=torch.hann_window(s, device=a.device), center=True, return_complex=True)
        sa, sb = torch.stft(a, **kw).abs(), torch.stft(b, **kw).abs()
        loss = loss + (sa - sb).abs().mean() * lin_weight \
                    + (torch.log(sa + 1e-7) - torch.log(sb + 1e-7)).abs().mean() * log_weight
    return loss
