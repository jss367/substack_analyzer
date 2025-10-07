from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from substack_analyzer.detection import detect_change_points

Effect = Literal["Transient", "Persistent", "No effect"]
Component = Literal["pulse", "level", "rate", "mixed", "none"]


@dataclass(frozen=True)
class BreakpointEffect:
    index: int
    date: pd.Timestamp
    effect: Effect
    component: Component
    slope_pre: float
    slope_post: float
    slope_delta: float
    jump_size: float  # ΔS spike at k
    z_spike: float
    note: str | None = None


def _mad(x: np.ndarray) -> float:
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def _fit_line(y: pd.Series) -> Tuple[float, float]:
    n = len(y)
    if n < 2:
        return 0.0, float(y.iloc[-1]) if n else 0.0
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, y.to_numpy(dtype=float), deg=1)  # slope, intercept
    return float(b), float(a)


def classify_breakpoints_effect(
    s: pd.Series,
    candidates: List[int],
    window: int = 6,
    z_pulse: float = 3.0,
    rate_factor: float = 0.75,
    level_factor: float = 0.75,
) -> List[BreakpointEffect]:
    """Return effect (Transient/Persistent/No effect) and component (pulse/level/rate/mixed/none) per candidate."""
    s = s.dropna().sort_index()
    if len(s) < (2 * window + 2):
        return []
    ds = s.diff().dropna()

    sig_delta = _mad(ds.to_numpy()) or (np.std(ds.to_numpy(), ddof=1) or 1.0)
    sig_level = _mad(s.to_numpy()) or (np.std(s.to_numpy(), ddof=1) or 1.0)
    tau_rate = rate_factor * sig_delta
    tau_level = level_factor * sig_level

    out: List[BreakpointEffect] = []
    for k in sorted(set(int(i) for i in candidates)):
        if k <= 0 or k >= len(s) - 1:
            continue
        pre_s = s.iloc[max(0, k - window) : k]
        post_s = s.iloc[k : min(len(s), k + window)]
        pre_d = ds.iloc[max(0, k - window) : k]
        post_d = ds.iloc[k : min(len(ds), k + window)]
        if len(pre_s) < 2 or len(post_s) < 2 or len(pre_d) < 1 or len(post_d) < 1:
            continue

        mu_pre, mu_post = float(np.median(pre_d)), float(np.median(post_d))
        delta_mu = mu_post - mu_pre
        spike = float(ds.iloc[k]) if k < len(ds) else 0.0
        z_spike = spike / (sig_delta or 1.0)

        slope_pre, a_pre = _fit_line(pre_s)
        slope_post, a_post = _fit_line(post_s)

        pred_pre_at_k = a_pre + slope_pre * len(pre_s)
        pred_post_at_k = a_post + slope_post * 0.0
        level_jump = float(s.iloc[k] - (pred_pre_at_k + pred_post_at_k) / 2.0)
        slope_delta = slope_post - slope_pre

        # Decide effect & component
        effect: Effect = "No effect"
        component: Component = "none"
        note = None

        if abs(z_spike) >= z_pulse and abs(delta_mu) < 0.5 * tau_rate:
            effect, component = "Transient", "pulse"
            note = f"pulse z≈{z_spike:.1f}"
        else:
            rate_flag = abs(slope_delta) >= tau_rate
            level_flag = abs(level_jump) >= tau_level
            if rate_flag and level_flag:
                effect, component = "Persistent", "mixed"
                note = f"Δslope={slope_delta:.3f}/mo; step≈{level_jump:.1f}"
            elif rate_flag:
                effect, component = "Persistent", "rate"
                note = f"Δslope={slope_delta:.3f}/mo"
            elif level_flag:
                effect, component = "Persistent", "level"
                note = f"step≈{level_jump:.1f}"
            else:
                effect, component = "No effect", "none"
                note = "weak/no change"

        out.append(
            BreakpointEffect(
                index=k,
                date=s.index[k].to_period("M").to_timestamp("M"),
                effect=effect,
                component=component,
                slope_pre=float(slope_pre),
                slope_post=float(slope_post),
                slope_delta=float(slope_delta),
                jump_size=float(spike),
                z_spike=float(z_spike),
                note=note,
            )
        )
    return out


def breakpoints_to_events(bps: List[BreakpointEffect], target_label: str) -> pd.DataFrame:
    rows = []
    for b in bps:
        if b.effect == "No effect":
            continue
        rows.append(
            {
                "date": b.date.date(),
                "type": "Other",  # keep your existing taxonomy
                "persistence": b.effect,
                "notes": f"{b.component}; {b.note} in {target_label}" if b.note else f"{b.component} in {target_label}",
                "cost": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["date", "type", "persistence", "notes", "cost"])


def breakpoints_for_segments(bps: List[BreakpointEffect]) -> List[int]:
    # Only rate/mixed imply an r change; those set the piecewise segments
    return sorted(set(b.index for b in bps if b.effect == "Persistent" and b.component in {"rate", "mixed"}))


def detect_and_classify(
    s: pd.Series,
    *,
    # detection knobs (mirror current detector defaults)
    max_changes: int = 4,
    min_seg_len: int = 2,
    penalty_scale: float = 4.0,
    # classification knobs
    window: int = 6,
    z_pulse: float = 3.0,
    rate_factor: float = 0.75,
    level_factor: float = 0.75,
    # optional override: provide explicit candidates
    candidates: Optional[List[int]] = None,
) -> List[BreakpointEffect]:
    """Detect candidate change points and classify their effects/components in one call."""
    s = s.dropna().sort_index()
    if candidates is None:
        candidates = detect_change_points(
            s,
            max_changes=max_changes,
            min_seg_len=min_seg_len,
            penalty_scale=penalty_scale,
            return_timestamps=False,
        )
    return classify_breakpoints_effect(
        s,
        candidates=candidates or [],
        window=window,
        z_pulse=z_pulse,
        rate_factor=rate_factor,
        level_factor=level_factor,
    )
