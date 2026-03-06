"""
Anchor-based xT Initialization for Neighbor Slots
==================================================
思路：推理时 xT 的 neighbor 未来帧不再用纯随机噪声，而是用
    x_{t_start} = alpha_t * anchor_normalized + sigma_t * randn
来初始化，给 DPM-Solver 提供更有意义的起点。

ego 槽位保持原始流程（route_lanes + guidance 已经足够引导）。
DPM-Solver 仍然是单路，完全不需要改变采样器本身。
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


# ──────────────────────────────────────────────────────────────────────────────
# 速度桶配置（与 kkmeans.py 保持一致）
# ──────────────────────────────────────────────────────────────────────────────
_SPEED_BINS = {
    "low":  (0.5,  5.0),
    "mid":  (5.0, 10.0),
    "high": (10.0, 9999.0),
}
_ANCHOR_HORIZON = 5.0   # anchor 时间跨度 (s)
_ANCHOR_DT      = 0.5   # anchor 采样间隔 (s)
_ANCHOR_T       = int(_ANCHOR_HORIZON / _ANCHOR_DT)   # = 10 个点


# ──────────────────────────────────────────────────────────────────────────────
# 加载 anchor 文件（懒加载，全局缓存）
# ──────────────────────────────────────────────────────────────────────────────
_anchor_cache: Optional[Dict[str, torch.Tensor]] = None

def _load_anchors(npz_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    global _anchor_cache
    if _anchor_cache is None:
        data = np.load(npz_path)
        _anchor_cache = {k: torch.tensor(data[k], dtype=torch.float32) for k in data.files}
    return {k: v.to(device) for k, v in _anchor_cache.items()}


def _select_bucket(speed: float) -> str:
    """根据速度标量选速度桶，找不到则返回 'low'。"""
    for name, (lo, hi) in _SPEED_BINS.items():
        if lo <= speed < hi:
            return name
    return "low"


def _temporal_interpolate(
    pts: torch.Tensor,   # (T_src,) 单条轨迹的一个分量
    T_dst: int,
    dt_src: float,
    dt_dst: float,
) -> torch.Tensor:
    """将单条 anchor 的 T_src 个点线性插值到 T_dst 个点，超出范围用末端速度外推。"""
    device = pts.device
    T_src = pts.shape[0]

    t_src = torch.arange(1, T_src + 1, dtype=torch.float32, device=device) * dt_src
    t_dst = torch.arange(1, T_dst + 1, dtype=torch.float32, device=device) * dt_dst

    # (1, 1, T_src) → (1, 1, T_dst)
    pts_interp = F.interpolate(
        pts.reshape(1, 1, T_src).float(),
        size=T_dst,
        mode='linear',
        align_corners=False,
    ).reshape(T_dst)

    # 超出 anchor 末尾的部分用末端速度外推
    t_max = t_src[-1]
    if t_dst[-1] > t_max:
        v_end = (pts[-1] - pts[-2]) / dt_src
        mask = t_dst > t_max
        pts_interp[mask] = pts[-1] + v_end * (t_dst[mask] - t_max)

    return pts_interp


def _anchor_to_normalized_single(
    anchor_lxly: torch.Tensor,   # (T_anchor*2,) 单条 anchor，lx,ly 交替
    future_len: int,
    state_normalizer,            # StateNormalizer
    slot_idx: int,               # 0=ego, 1+=neighbor，用于选归一化参数
    device: torch.device,
) -> torch.Tensor:
    """
    将单条 anchor (lx,ly) 转为归一化的 (x,y,cos,sin) 序列。
    返回 shape: ((1+future_len)*4,)，含 t=0 的当前帧（全零，代表原点）。
    """
    lx = anchor_lxly[0::2]   # (T_anchor,)
    ly = anchor_lxly[1::2]   # (T_anchor,)

    lx_i = _temporal_interpolate(lx, future_len, _ANCHOR_DT, 0.1)   # (future_len,)
    ly_i = _temporal_interpolate(ly, future_len, _ANCHOR_DT, 0.1)

    # 拼上 t=0 的当前帧 (0,0)
    x_seq = torch.cat([torch.zeros(1, device=device), lx_i])   # (1+future_len,)
    y_seq = torch.cat([torch.zeros(1, device=device), ly_i])

    # 用差分估算朝向
    dx = x_seq[1:] - x_seq[:-1]   # (future_len,)
    dy = y_seq[1:] - y_seq[:-1]
    norm = torch.clamp(torch.sqrt(dx**2 + dy**2), min=1e-6)
    cos_h = dx / norm
    sin_h = dy / norm

    # t=0 帧的朝向用第一段差分方向
    cos_seq = torch.cat([cos_h[:1], cos_h])   # (1+future_len,)
    sin_seq = torch.cat([sin_h[:1], sin_h])

    # 拼成 (1+future_len, 4)
    traj = torch.stack([x_seq, y_seq, cos_seq, sin_seq], dim=-1)

    # 归一化：用对应 slot 的 mean/std
    mean = state_normalizer.mean[slot_idx].to(device)   # (1, 4)
    std  = state_normalizer.std[slot_idx].to(device)    # (1, 4)
    traj_norm = (traj - mean) / std

    return traj_norm.reshape(-1)   # ((1+future_len)*4,)


def build_anchor_xT(
    current_states: torch.Tensor,      # (B, P, 4) ego+neighbors 当前状态（原始，非归一化）
    neighbor_speeds: torch.Tensor,     # (B, P-1) 每个 neighbor 的当前速度 (m/s)
    neighbor_current_mask: torch.Tensor,  # (B, P-1) True=无效/padding
    future_len: int,
    state_normalizer,
    anchor_npz_path: str,
    t_start: float = 0.5,             # 截断时刻，0~1
) -> torch.Tensor:
    """
    构建 anchor 初始化的 xT，用于替换原始的纯随机初始化。

    - ego 槽位（slot 0）：t=0 帧锚定为 current_state，未来帧用随机噪声（与原始相同）
    - neighbor 槽位（slot 1~P-1）：
        - 无效（padding）neighbor：随机噪声（与原始相同）
        - 有效 neighbor：根据速度选 anchor，加截断噪声 x = alpha_t * anchor + sigma_t * randn

    Returns:
        xT: (B, P, (1+future_len)*4)，可直接替换原始的 xT 送入 dpm_sampler
    """
    B, P, _ = current_states.shape
    device = current_states.device
    D = (1 + future_len) * 4

    anchors_dict = _load_anchors(anchor_npz_path, device)

    # 计算 t_start 对应的 alpha, sigma
    noise_schedule = dpm.NoiseScheduleVP(schedule='linear')
    t_vec = torch.full((1,), t_start, device=device)
    alpha_t = noise_schedule.marginal_alpha(t_vec).item()
    sigma_t = noise_schedule.marginal_std(t_vec).item()

    # ── 初始化 xT：ego 槽位用 current_state 拼随机噪声（原始逻辑）────
    # shape: (B, P, 1+future_len, 4)
    xT = torch.cat([
        current_states[:, :, None, :],                              # (B, P, 1, 4)  当前帧
        torch.randn(B, P, future_len, 4, device=device) * 0.5,     # (B, P, T, 4)  随机未来帧
    ], dim=2)

    # ── 对有效 neighbor 槽位用 anchor 替换未来帧 ──────────────────────
    for b in range(B):
        for n in range(P - 1):               # n 是 neighbor 索引（0-based in neighbor 维）
            p = n + 1                         # p 是 xT 中的 slot 索引
            if neighbor_current_mask[b, n]:  # True = 无效/padding，跳过
                continue

            # 根据该 neighbor 的速度选 anchor 桶，随机选一条 anchor
            speed = neighbor_speeds[b, n].item()
            bucket = _select_bucket(speed)
            anchor_pool = anchors_dict.get(bucket,
                          anchors_dict.get("low",
                          list(anchors_dict.values())[0]))  # (K, T_anchor*2)
            K = anchor_pool.shape[0]
            k = torch.randint(0, K, (1,)).item()
            anchor_single = anchor_pool[k]   # (T_anchor*2,)

            # anchor → 归一化的 (1+future_len, 4) 序列
            anchor_norm = _anchor_to_normalized_single(
                anchor_single, future_len, state_normalizer,
                slot_idx=min(p, state_normalizer.mean.shape[0] - 1),
                device=device,
            ).reshape(1 + future_len, 4)   # (1+future_len, 4)

            # 只替换未来帧部分（index 1 以后），用截断加噪
            anchor_future = anchor_norm[1:]   # (future_len, 4)
            noise = torch.randn_like(anchor_future)
            x_future = alpha_t * anchor_future + sigma_t * noise   # (future_len, 4)

            xT[b, p, 1:] = x_future   # 替换未来帧，t=0 帧仍保持 current_state

    return xT.reshape(B, P, -1)   # (B, P, (1+future_len)*4)
