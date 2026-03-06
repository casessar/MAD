"""
Truncated Diffusion Visualization — 16 Real Anchors
=====================================================

Pipeline:
  anchor_k  (clean trajectory, k=0..15)
    |
    |  Standard Diffusion  : add_noise(anchor_k, t=1.0)  → pure Gaussian noise
    |  Truncated Diffusion : add_noise(anchor_k, t=0.5)  → anchor-centered noise
    ↓
  x_noised
    → simulate denoising  → x_0
    → pick best candidate (closest to a simulated GT)

Layout (single figure, 4 rows × 5 cols):
  Row 0: Standard  Diffusion — noised trajectories (16 anchors overlaid)
  Row 1: Standard  Diffusion — denoised results    (16 anchors overlaid) + best
  Row 2: Truncated Diffusion — noised trajectories (16 anchors overlaid)
  Row 3: Truncated Diffusion — denoised results    (16 anchors overlaid) + best

  Columns per row:
    col 0 : all  16 candidates overlaid
    col 1 : low-speed group   (final_x ≤ 15)
    col 2 : mid-speed group   (15 < final_x ≤ 35)
    col 3 : high-speed group  (final_x > 35)
    col 4 : BEST candidate only
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load real anchors  (16 × 10 × 2)
# ─────────────────────────────────────────────────────────────────────────────
_d        = np.load("vis_output/anchors_all.npz")
_motion   = _d["all"].reshape(15, 10, 2)     # (15,10,2)
_static   = _d["static"].reshape(1, 10, 2)   # (1,10,2)
ANCHORS   = np.concatenate([_motion, _static], axis=0)   # (16,10,2)
N_ANC     = 16
FUTURE_LEN = 10

# Speed groups for column partitioning
def _speed_group(a):
    fx = a[-1, 0]
    if fx <= 15:   return "low"
    elif fx <= 35: return "mid"
    else:          return "high"

GROUP_IDX = {"low": [], "mid": [], "high": []}
for _i, _a in enumerate(ANCHORS):
    GROUP_IDX[_speed_group(_a)].append(_i)

# ─────────────────────────────────────────────────────────────────────────────
# 1. VPSDE  marginal_prob
# ─────────────────────────────────────────────────────────────────────────────
BETA_MIN, BETA_MAX = 0.1, 20.0

def marginal_prob(x: torch.Tensor, t: float):
    t_ = torch.tensor(t, dtype=torch.float32)
    log_coeff = -0.25 * t_**2 * (BETA_MAX - BETA_MIN) - 0.5 * BETA_MIN * t_
    mean = torch.exp(log_coeff) * x
    std  = torch.sqrt(torch.clamp(1 - torch.exp(2.0 * log_coeff), min=1e-6))
    return mean, std

def add_noise(x: torch.Tensor, t: float, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    mean, std = marginal_prob(x, t)
    return mean + std * torch.randn_like(x)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulated denoiser
#    - For Standard  (t_start=1.0): gradually recovers anchor shape
#    - For Truncated (t_start=0.5): starts closer to anchor → less drift
# ─────────────────────────────────────────────────────────────────────────────
def fake_denoise(x_noised: torch.Tensor,
                 anchor:   torch.Tensor,
                 t_start:  float,
                 steps:    int = 20,
                 seed:     int = 0) -> tuple:
    """Returns (x_final, list_of_intermediate_tensors)."""
    torch.manual_seed(seed)
    x = x_noised.clone()
    traj = [x.clone()]
    ts = torch.linspace(t_start, 0.0, steps + 1)
    for i in range(steps):
        t_cur  = ts[i].item()
        t_next = ts[i + 1].item()
        alpha  = (t_cur - t_next) / max(t_start, 1e-6)
        # Residual stochasticity: proportional to remaining time
        noise_scale = (t_next / max(t_start, 1e-6)) * 0.25
        x = x + alpha * (anchor - x) + noise_scale * torch.randn_like(x)
        traj.append(x.clone())
    return x, traj

# ─────────────────────────────────────────────────────────────────────────────
# 3. Simulated GT trajectory (mid-speed, slight left curve)
# ─────────────────────────────────────────────────────────────────────────────
_t  = torch.linspace(0.0, 1.0, FUTURE_LEN)
GT  = torch.stack([_t * 28.0, _t ** 1.5 * 4.0], dim=-1)   # (10,2)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run both diffusion schemes across all 16 anchors
# ─────────────────────────────────────────────────────────────────────────────
def run_all(t_start: float, denoise_steps: int):
    anchors_t = torch.tensor(ANCHORS, dtype=torch.float32)   # (16,10,2)
    noised, denoised, intermed = [], [], []
    for k in range(N_ANC):
        anc   = anchors_t[k]
        xn    = add_noise(anc, t_start, seed=k * 17 + int(t_start * 100))
        x0, traj = fake_denoise(xn, anc, t_start,
                                 steps=denoise_steps,
                                 seed=k * 31 + int(t_start * 100))
        noised.append(xn)
        denoised.append(x0)
        intermed.append(traj)

    # Best = lowest L2 to GT
    losses   = [((d - GT) ** 2).sum().item() for d in denoised]
    best_idx = int(np.argmin(losses))
    return noised, denoised, intermed, losses, best_idx

torch.manual_seed(42)
noised_S, denois_S, inter_S, loss_S, best_S = run_all(t_start=1.0, denoise_steps=20)
noised_T, denois_T, inter_T, loss_T, best_T = run_all(t_start=0.5, denoise_steps=10)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Color palette: one color per anchor (16 distinct colors)
# ─────────────────────────────────────────────────────────────────────────────
_cmap   = get_cmap("tab20")
COLORS  = [_cmap(i / 16) for i in range(N_ANC)]
# Static anchor (index 15) → grey
COLORS[15] = (0.65, 0.65, 0.65, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure layout  (4 rows × 5 cols)
# ─────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 26, 20
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("#12121e")

ROW_LABELS = [
    "Standard  (t=1.0) — Noised",
    "Standard  (t=1.0) — Denoised",
    "Truncated (t=0.5) — Noised",
    "Truncated (t=0.5) — Denoised",
]
COL_LABELS = [
    "All 16 Candidates",
    "Low-Speed Group\n(final x ≤ 15 m)",
    "Mid-Speed Group\n(15 < x ≤ 35 m)",
    "High-Speed Group\n(x > 35 m)",
    "★ Best Candidate",
]

axes = [[fig.add_subplot(4, 5, r * 5 + c + 1)
         for c in range(5)] for r in range(4)]

# Row-label text on the left
for r in range(4):
    axes[r][0].set_ylabel(ROW_LABELS[r], color="white",
                          fontsize=9, fontweight="bold",
                          labelpad=8, rotation=90, va="center")

# Column titles on top
for c in range(5):
    axes[0][c].set_title(COL_LABELS[c], color="#CCCCEE",
                         fontsize=8, fontweight="bold", pad=6)

def _style_ax(ax, xlim=(-5, 75), ylim=(-28, 28)):
    ax.set_facecolor("#0c0c1a")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(colors="#666688", labelsize=6)
    ax.grid(True, color="#22223a", linewidth=0.5, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

def _draw_road(ax):
    ax.axhline(y= 3.75, color="#444466", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(y=-3.75, color="#444466", lw=0.8, ls="--", alpha=0.6)

def _draw_ego(ax):
    rect = mpatches.FancyBboxPatch((-1.1, -0.55), 2.2, 1.1,
                                    boxstyle="round,pad=0.1",
                                    facecolor="#FFD700", edgecolor="white",
                                    linewidth=1.2, zorder=6)
    ax.add_patch(rect)
    ax.text(0, 0, "EGO", ha="center", va="center",
            fontsize=5, color="#1a1a2e", fontweight="bold", zorder=7)

def _draw_gt(ax):
    ax.plot(GT[:, 0].numpy(), GT[:, 1].numpy(),
            color="#00FF88", lw=2.0, ls=":", alpha=0.85,
            zorder=5, label="GT")

def _draw_anchor(ax, k, alpha=0.45):
    a = ANCHORS[k]
    ax.plot(a[:, 0], a[:, 1], "--", color=COLORS[k],
            lw=1.0, alpha=alpha, zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Draw helper — fills one ax with trajectories for given anchor indices
# ─────────────────────────────────────────────────────────────────────────────
def fill_ax(ax, idx_list, traj_list, t_start, best_idx,
            show_gt=True, highlight_best=True, lw=1.4, alpha=0.75):
    _style_ax(ax)
    _draw_road(ax)
    _draw_ego(ax)
    if show_gt:
        _draw_gt(ax)

    for k in idx_list:
        is_best = (k == best_idx) and highlight_best
        traj = traj_list[k]
        lw_k = 2.5 if is_best else lw
        a_k  = 1.0 if is_best else alpha
        zo   = 5  if is_best else 4
        ax.plot(traj[:, 0].numpy(), traj[:, 1].numpy(),
                color=COLORS[k], lw=lw_k, alpha=a_k, zorder=zo)
        # Mark anchor mean with dashed line
        _draw_anchor(ax, k, alpha=0.3)

    if highlight_best and (best_idx in idx_list):
        bx = traj_list[best_idx]
        ax.plot(bx[-1, 0].item(), bx[-1, 1].item(),
                "*", color="#FFD700", markersize=12, zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Draw all 4 rows
# ─────────────────────────────────────────────────────────────────────────────
for row_idx, (noised_list, denois_list, t_start) in enumerate([
    (noised_S, None,     1.0),   # row 0: standard noised
    (None,     denois_S, 1.0),   # row 1: standard denoised
    (noised_T, None,     0.5),   # row 2: truncated noised
    (None,     denois_T, 0.5),   # row 3: truncated denoised
]):
    is_noised = (noised_list is not None)
    traj_src  = noised_list if is_noised else denois_list
    best_idx  = best_S if t_start == 1.0 else best_T

    for col_idx, (group_key, idx_list) in enumerate([
        ("all",  list(range(N_ANC))),
        ("low",  GROUP_IDX["low"]),
        ("mid",  GROUP_IDX["mid"]),
        ("high", GROUP_IDX["high"]),
        ("best", [best_idx]),
    ]):
        ax = axes[row_idx][col_idx]
        _style_ax(ax)
        _draw_road(ax)
        _draw_ego(ax)
        _draw_gt(ax)

        show_list = idx_list if group_key != "best" else [best_idx]
        for k in show_list:
            traj = traj_src[k]
            is_best = (k == best_idx)
            ax.plot(traj[:, 0].numpy(), traj[:, 1].numpy(),
                    color=COLORS[k],
                    lw=2.8 if is_best else 1.3,
                    alpha=1.0 if is_best else (0.72 if is_noised else 0.80),
                    zorder=6 if is_best else 4)
            # Anchor dashed underlay
            _draw_anchor(ax, k, alpha=0.28 if not is_noised else 0.18)

            # Best star
            if is_best and col_idx == 4:
                ax.plot(traj[-1, 0].item(), traj[-1, 1].item(),
                        "*", color="#FFD700", markersize=14, zorder=9,
                        label=f"Best (k={best_idx})")

        # Noise cloud annotation in noised rows
        if is_noised and col_idx == 0:
            ax.text(0.02, 0.97,
                    f"t_start = {t_start:.1f}\nσ = {0.95 if t_start==1.0 else 0.55:.2f}",
                    transform=ax.transAxes, fontsize=7, color="#FF9966",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#1a1a2e", ec="#FF9966", alpha=0.8))

        if col_idx == 4 and not is_noised:
            loss_src = loss_S if t_start == 1.0 else loss_T
            ax.text(0.02, 0.97,
                    f"Anchor k={best_idx}\nL2={loss_src[best_idx]:.2f}",
                    transform=ax.transAxes, fontsize=7, color="#FFD700",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#1a1a2e", ec="#FFD700", alpha=0.8))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Shared legend for the 16 anchors (bottom of figure)
# ─────────────────────────────────────────────────────────────────────────────
speed_tags = []
for k in range(N_ANC):
    fx = ANCHORS[k, -1, 0]
    fy = ANCHORS[k, -1, 1]
    if k == 15:
        tag = "Static"
    elif abs(fy) > 8:
        side = "L" if fy > 0 else "R"
        tag  = f"Turn-{side}"
    elif fx > 40:
        tag = f"Hi {fx:.0f}m"
    elif fx > 15:
        tag = f"Mid {fx:.0f}m"
    else:
        tag = f"Lo {fx:.0f}m"
    speed_tags.append(tag)

legend_handles = [
    Line2D([0], [0], color=COLORS[k], lw=2.0, label=f"[{k:02d}] {speed_tags[k]}")
    for k in range(N_ANC)
]
legend_handles += [
    Line2D([0], [0], color="#00FF88", lw=1.8, ls=":", label="GT trajectory"),
    Line2D([0], [0], color="white",   lw=0,   marker="*",
           markersize=10, markerfacecolor="#FFD700", label="Best candidate"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=9, fontsize=7.5,
           facecolor="#1a1a2e", edgecolor="#444466",
           labelcolor="white", framealpha=0.9,
           bbox_to_anchor=(0.5, 0.0))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Divider line between Standard and Truncated rows
# ─────────────────────────────────────────────────────────────────────────────
fig.add_artist(
    plt.Line2D([0.01, 0.99], [0.505, 0.505],
               transform=fig.transFigure,
               color="#FF6644", lw=1.5, ls="--", alpha=0.7)
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Super-title and row banner texts
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Truncated Diffusion Planner  —  16 Real K-Means Anchors\n"
    "Standard (t: 1.0→0) vs Truncated (t: 0.5→0)",
    fontsize=14, color="white", fontweight="bold", y=1.005
)

for r, (label, ypos, col) in enumerate([
    ("◀ STANDARD DIFFUSION  (t_start = 1.0) ▶", 0.765, "#5599FF"),
    ("◀ TRUNCATED DIFFUSION (t_start = 0.5) ▶", 0.255, "#FF7744"),
]):
    fig.text(0.5, ypos, label,
             ha="center", va="center", fontsize=11, color=col,
             fontweight="bold", alpha=0.85)

plt.subplots_adjust(left=0.07, right=0.99, top=0.95,
                    bottom=0.085, hspace=0.35, wspace=0.28)

out = "vis_output/truncated_diffusion_demo.png"
plt.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
print(f"\nStandard  best anchor : k={best_S}  ({speed_tags[best_S]})  L2={loss_S[best_S]:.3f}")
print(f"Truncated best anchor : k={best_T}  ({speed_tags[best_T]})  L2={loss_T[best_T]:.3f}")
