from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_planner.utils.normalizer import StateNormalizer

# ── Anchor 全局缓存（训练阶段共享） ──────────────────────────────────────────
_LOSS_ANCHOR_CACHE: dict = {}

# 速度桶定义（与 kkmeans.py 保持一致）
_SPEED_BINS_LOSS = [("low", 0.0, 5.0), ("mid", 5.0, 10.0), ("high", 10.0, 9999.0)]


def _load_anchors_for_loss(npz_path: str, device: torch.device) -> dict:
    """加载所有桶的 anchor，返回 {bucket_name: Tensor(K, 20)} 并缓存。"""
    global _LOSS_ANCHOR_CACHE
    if npz_path not in _LOSS_ANCHOR_CACHE:
        data = np.load(npz_path)
        _LOSS_ANCHOR_CACHE[npz_path] = {
            k: torch.tensor(data[k], dtype=torch.float32) for k in data.files
        }
    return {k: v.to(device) for k, v in _LOSS_ANCHOR_CACHE[npz_path].items()}


def _select_anchors_by_speed(
    ego_current: torch.Tensor,   # (B, 4): [x, y, cos_h, sin_h]
    anchors_dict: dict,          # {bucket_name: Tensor(K, 20)}
    device: torch.device,
) -> torch.Tensor:
    """退化为随机选（ego_current 无速度维度时使用）"""
    all_anchors = torch.cat(list(anchors_dict.values()), dim=0)  # (K_total, 20)
    K_total = all_anchors.shape[0]
    B = ego_current.shape[0]
    idxs = torch.randint(0, K_total, (B,), device=device)
    return all_anchors[idxs]  # (B, 20)


def _select_anchors_by_speed_v2(
    ego_speed: torch.Tensor,     # (B,): 每个样本的当前速度（m/s）
    anchors_dict: dict,          # {bucket_name: Tensor(K, 20)}
    device: torch.device,
) -> torch.Tensor:
    """
    按真实速度从对应速度桶选 anchor（向量化实现，无 Python for 循环）。
    策略：
      1. 将所有桶 anchor 拼接为 (K_total, 20)；
      2. 用速度区间端点向量化判断每个样本所属桶；
      3. 对每个桶内的样本用 randint 批量选索引，再 gather。
    返回: (B, 20)
    """
    B = ego_speed.shape[0]

    # 拼接所有桶，并记录每桶的起止偏移
    bucket_tensors = []
    bucket_ranges = {}   # name -> (start, end)
    offset = 0
    for name, lo, hi in _SPEED_BINS_LOSS:
        if name in anchors_dict:
            t = anchors_dict[name].to(device)   # (K, 20)
            K = t.shape[0]
            bucket_tensors.append(t)
            bucket_ranges[name] = (offset, offset + K)
            offset += K
    all_anchors = torch.cat(bucket_tensors, dim=0)  # (K_total, 20)
    K_total = offset

    # 为每个样本确定桶内随机索引（向量化）
    global_idxs = torch.randint(0, K_total, (B,), device=device)   # 默认：全局随机

    spd = ego_speed.to(device)
    for name, lo, hi in _SPEED_BINS_LOSS:
        if name not in bucket_ranges:
            continue
        start, end = bucket_ranges[name]
        K = end - start
        mask = (spd >= lo) & (spd < hi)           # (B,) bool
        n = int(mask.sum().item())
        if n > 0:
            local_idxs = torch.randint(0, K, (n,), device=device) + start
            global_idxs[mask] = local_idxs

    return all_anchors[global_idxs]   # (B, 20)


def _build_ego_anchor_xT(
    ego_current: torch.Tensor,    # (B, 4): [x, y, cos_h, sin_h]
    ego_speed: Optional[torch.Tensor],  # (B,): 真实速度（m/s），可为 None
    future_len: int,
    norm: StateNormalizer,
    anchor_npz_path: str,
    t: torch.Tensor,
    marginal_prob: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为训练 ego 槽位构造 anchor 加噪的 xT。
    返回:
      xT:      (B, 1+T, 4)  归一化空间（含 t=0 帧）
      noise:   (B, T, 4)    真实 ε（仅 future 帧）
      ego_std: (B, 1, 1)    对应的 std_coeff，用于 loss 计算
    """
    B = ego_current.shape[0]
    device = ego_current.device
    anchors_dict = _load_anchors_for_loss(anchor_npz_path, device)

    # 按速度桶选 anchor（向量化）
    if ego_speed is not None:
        anchors = _select_anchors_by_speed_v2(ego_speed, anchors_dict, device)
    else:
        anchors = _select_anchors_by_speed(ego_current, anchors_dict, device)
    # anchors: (B, 20)

    lx = anchors[:, 0::2]  # (B, 10)
    ly = anchors[:, 1::2]  # (B, 10)
    lx_i = F.interpolate(
        lx.unsqueeze(1), size=future_len, mode='linear', align_corners=False
    ).squeeze(1)
    ly_i = F.interpolate(
        ly.unsqueeze(1), size=future_len, mode='linear', align_corners=False
    ).squeeze(1)

    x_seq = torch.cat([torch.zeros(B, 1, device=device), lx_i], dim=1)
    y_seq = torch.cat([torch.zeros(B, 1, device=device), ly_i], dim=1)
    dx = x_seq[:, 1:] - x_seq[:, :-1]
    dy = y_seq[:, 1:] - y_seq[:, :-1]
    nrm = torch.clamp(torch.sqrt(dx ** 2 + dy ** 2), min=1e-6)
    cos_h = dx / nrm
    sin_h = dy / nrm
    cos_seq = torch.cat([cos_h[:, :1], cos_h], dim=1)
    sin_seq = torch.cat([sin_h[:, :1], sin_h], dim=1)

    traj = torch.stack([x_seq, y_seq, cos_seq, sin_seq], dim=-1)  # (B, 1+T, 4)

    mean_n = norm.mean[0].to(device)  # (1, 4)
    std_n  = norm.std[0].to(device)   # (1, 4)
    traj_norm = (traj - mean_n) / std_n

    cur_norm = (ego_current - mean_n[0]) / std_n[0]
    traj_norm[:, 0, :] = cur_norm

    future_norm = traj_norm[:, 1:, :]  # (B, T, 4)
    mean_coeff, std_coeff = marginal_prob(future_norm, t)
    # std_coeff shape: (B,) 或 (B,1) → 整理为 (B,1,1)
    std_v = std_coeff.view(B, *([1] * (len(future_norm.shape) - 1)))  # (B,1,1)
    noise = torch.randn_like(future_norm)
    future_xT = mean_coeff + std_v * noise

    xT = torch.cat([traj_norm[:, :1, :], future_xT], dim=1)  # (B, 1+T, 4)
    return xT, noise, std_v  # std_v: (B,1,1)


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    eps: float = 1e-3,
    t_start: float = 1.0,
    anchor_npz_path: Optional[str] = None,
):
    ego_future, neighbors_future, neighbor_future_mask = futures

    B, Pn, T, _ = neighbors_future.shape
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1
    )

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    P = gt_future.shape[1]

    # ── 截断扩散：t ∈ (eps, t_start) ────────────────────────────────────────
    t = torch.rand(B, device=gt_future.device) * (t_start - eps) + eps
    z = torch.randn_like(gt_future, device=gt_future.device)

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean_c, std_c = marginal_prob(all_gt[..., 1:, :], t)
    std_v = std_c.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))  # (B,1,1,1)

    xT = mean_c + std_v * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    # ── Ego anchor 初始化：替换 ego 槽位为 anchor 加噪 ──────────────────────
    ego_std_v = None  # 用于 ego 槽位的独立 std_v
    if anchor_npz_path is not None:
        # 尝试从 inputs 获取真实速度（如 ego_current_state 有更多维度）
        ego_speed = None
        ego_state_full = inputs.get("ego_current_state", None)
        if ego_state_full is not None and ego_state_full.shape[-1] >= 6:
            vx = ego_state_full[:, 4]
            vy = ego_state_full[:, 5]
            ego_speed = torch.sqrt(vx ** 2 + vy ** 2)  # (B,)

        ego_xT, ego_noise, ego_std_v = _build_ego_anchor_xT(
            ego_current=current_states[:, 0, :],
            ego_speed=ego_speed,
            future_len=T,
            norm=norm,
            anchor_npz_path=anchor_npz_path,
            t=t,
            marginal_prob=marginal_prob,
        )
        xT[:, 0, :, :] = ego_xT          # (B, 1+T, 4)
        z[:, 0, :, :] = ego_noise         # (B, T, 4)

    # ── EGO-ONLY：neighbors 未来帧置零（与推理对齐） ─────────────────────────
    xT[:, 1:, 1:, :] = 0.0

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs)
    score = decoder_output["score"][:, :, 1:, :]  # [B, P, T, 4]

    if model_type == "score":
        if ego_std_v is not None:
            # ── ego 槽位：用自己的 std_v（anchor 加噪的 std）计算 loss ────────
            # ego_std_v: (B,1,1)；score[:,0]: (B,T,4)；z[:,0]: (B,T,4)
            ego_loss_raw = (score[:, 0, :, :] * ego_std_v + z[:, 0, :, :]) ** 2
            dpm_loss_ego = ego_loss_raw.sum(dim=-1)  # (B,T)

            # ── neighbors 槽位：全零未来帧，无有效监督信号
            # 用 detach() 切断梯度，防止全零轨迹 score 给网络引入有害梯度
            if P > 1:
                neigh_score = score[:, 1:, :, :].detach()   # (B,P-1,T,4)，梯度切断
                neigh_z     = z[:, 1:, :, :]                # (B,P-1,T,4)
                dpm_loss_neigh = (neigh_score * std_v + neigh_z).pow(2).sum(dim=-1)  # (B,P-1,T)
                dpm_loss = torch.cat(
                    [dpm_loss_ego.unsqueeze(1), dpm_loss_neigh], dim=1
                )  # (B,P,T)
            else:
                dpm_loss = dpm_loss_ego.unsqueeze(1)  # (B,1,T)
        else:
            dpm_loss = torch.sum((score * std_v + z) ** 2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :]) ** 2, dim=-1)

    # ── neighbors 未来全零，无有效监督，neighbor_prediction_loss 固定为 0 ──────
    loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=gt_future.device)
    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    assert not torch.isnan(dpm_loss[:, 0, :]).any(), \
        f"ego loss cannot be nan, z={z[:, 0]}"

    return loss, decoder_output