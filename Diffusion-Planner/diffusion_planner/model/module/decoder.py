import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer

# ── Anchor 全局缓存 ────────────────────────────────────────────────────────────
_ANCHOR_CACHE: dict = {}

def _load_anchors_ego(npz_path: str, device: torch.device) -> dict:
    """加载并缓存 anchor，返回 {bucket: Tensor(K, 20)}"""
    global _ANCHOR_CACHE
    if npz_path not in _ANCHOR_CACHE:
        data = np.load(npz_path)
        _ANCHOR_CACHE[npz_path] = {k: torch.tensor(data[k], dtype=torch.float32) for k in data.files}
    return {k: v.to(device) for k, v in _ANCHOR_CACHE[npz_path].items()}


_SPEED_BINS = [("low", 0.0, 5.0), ("mid", 5.0, 10.0), ("high", 10.0, 9999.0)]


def _anchor_to_traj_norm_batch(
    anchors: torch.Tensor,          # (M, 20)
    future_len: int,
    ego_current_batch: torch.Tensor,  # (M, 4)
    state_normalizer,
    device: torch.device,
) -> torch.Tensor:
    """
    批量将 M 条 anchor (M, 20) 插值、归一化并替换 t=0 帧。
    返回: (M, 1+future_len, 4)
    """
    M = anchors.shape[0]
    lx = anchors[:, 0::2]  # (M, 10)
    ly = anchors[:, 1::2]  # (M, 10)

    # 批量插值：(M, 1, 10) -> (M, 1, future_len) -> (M, future_len)
    lx_i = F.interpolate(lx.unsqueeze(1), size=future_len, mode='linear', align_corners=False).squeeze(1)
    ly_i = F.interpolate(ly.unsqueeze(1), size=future_len, mode='linear', align_corners=False).squeeze(1)

    x_seq = torch.cat([torch.zeros(M, 1, device=device), lx_i], dim=1)  # (M, 1+T)
    y_seq = torch.cat([torch.zeros(M, 1, device=device), ly_i], dim=1)  # (M, 1+T)

    dx = x_seq[:, 1:] - x_seq[:, :-1]  # (M, T)
    dy = y_seq[:, 1:] - y_seq[:, :-1]  # (M, T)
    nrm = torch.clamp(torch.sqrt(dx ** 2 + dy ** 2), min=1e-6)
    cos_h = dx / nrm  # (M, T)
    sin_h = dy / nrm  # (M, T)
    cos_seq = torch.cat([cos_h[:, :1], cos_h], dim=1)  # (M, 1+T)
    sin_seq = torch.cat([sin_h[:, :1], sin_h], dim=1)  # (M, 1+T)

    traj = torch.stack([x_seq, y_seq, cos_seq, sin_seq], dim=-1)  # (M, 1+T, 4)

    mean = state_normalizer.mean[0].to(device)  # (1, 4)
    std  = state_normalizer.std[0].to(device)   # (1, 4)
    traj_norm = (traj - mean) / std              # (M, 1+T, 4)

    # t=0 帧批量替换为真实当前状态（归一化）
    cur_norm = (ego_current_batch - mean[0]) / std[0]  # (M, 4)
    traj_norm[:, 0, :] = cur_norm
    return traj_norm  # (M, 1+T, 4)


def _build_ego_xT_multi(
    ego_current: torch.Tensor,        # (B, 4): [x, y, cos_h, sin_h]
    future_len: int,
    state_normalizer,
    anchor_npz_path: str,
    num_anchors: int = 3,
    ego_speed: torch.Tensor = None,   # (B,) 真实速度(m/s)，可为 None
) -> torch.Tensor:
    """
    按速度桶选 num_anchors 条 anchor（与训练对齐），
    批量插值归一化后返回 (B, num_anchors, 1+future_len, 4)。
    完全向量化，无双重 Python 循环。
    """
    B = ego_current.shape[0]
    N = num_anchors
    device = ego_current.device
    anchors_dict = _load_anchors_ego(anchor_npz_path, device)

    # 拼接所有桶并记录偏移
    bucket_tensors, bucket_ranges = [], {}
    offset = 0
    for name, lo, hi in _SPEED_BINS:
        if name in anchors_dict:
            t_anc = anchors_dict[name]  # (K, 20)
            K = t_anc.shape[0]
            bucket_tensors.append(t_anc)
            bucket_ranges[name] = (offset, offset + K)
            offset += K
    all_anchors = torch.cat(bucket_tensors, dim=0)  # (K_total, 20)
    K_total = offset

    # ── 向量化生成 (B*N,) 全局 anchor 索引（消除外层 for n 循环） ────────────
    flat_idxs = torch.randint(0, K_total, (B * N,), device=device)   # 默认全局随机

    if ego_speed is not None:
        # 将速度复制 N 次 → (B*N,)，与 flat_idxs 对齐后向量化按桶替换
        spd_rep = ego_speed.to(device).repeat_interleave(N)  # (B*N,)
        for name, lo, hi in _SPEED_BINS:
            if name not in bucket_ranges:
                continue
            start, end = bucket_ranges[name]
            K = end - start
            mask = (spd_rep >= lo) & (spd_rep < hi)          # (B*N,) bool
            n_mask = int(mask.sum().item())
            if n_mask > 0:
                flat_idxs[mask] = torch.randint(0, K, (n_mask,), device=device) + start

    # (B*N, 20)
    anchors_flat = all_anchors[flat_idxs]
    # ego_current 重复 N 次 → (B*N, 4)
    ego_rep = ego_current.repeat_interleave(N, dim=0)

    # 批量插值 + 归一化（无 Python 循环）
    traj_flat = _anchor_to_traj_norm_batch(
        anchors_flat, future_len, ego_rep, state_normalizer, device
    )  # (B*N, 1+T, 4)

    return traj_flat.reshape(B, N, 1 + future_len, 4)    # (B, N, 1+T, 4)


def _select_best_candidate(
    x0_denorm: torch.Tensor,   # (B*N, P, 1+T, 4) 反归一化后
    B: int,
    N: int,
    w_jerk: float = 1.0,
    w_heading: float = 0.5,
    w_speed: float = 0.3,
    ego_speed_gt: torch.Tensor = None,  # (B,) 真实起始速度(m/s)，可为 None
    dt: float = 0.1,                    # 时间步长(s)，用于速度估算
) -> torch.Tensor:
    """
    综合加速度（jerk）平滑度、方向一致性、速度连续性选最优候选。
    若提供 ego_speed_gt，速度连续性得分使用真实当前速度对比（更精确）。
    返回 best_idx: (B,)
    """
    ego_traj = x0_denorm[:, 0, 1:, :]   # (B*N, T, 4): 去掉 t=0 帧

    # ── 1. Jerk（加速度差分）最小 ─────────────────────────────────────────────
    xy    = ego_traj[..., :2]                              # (B*N, T, 2)
    vel   = xy[:, 1:, :] - xy[:, :-1, :]                  # (B*N, T-1, 2)
    accel = vel[:, 1:, :] - vel[:, :-1, :]                # (B*N, T-2, 2)
    jerk_score = accel.pow(2).sum(dim=(-1, -2))            # (B*N,)

    # ── 2. 方向一致性：相邻帧 heading 变化量之和（越小越好） ──────────────────
    cos_h = ego_traj[..., 2]   # (B*N, T)
    sin_h = ego_traj[..., 3]   # (B*N, T)
    cos_delta = cos_h[:, 1:] * cos_h[:, :-1] + sin_h[:, 1:] * sin_h[:, :-1]  # (B*N, T-1)
    heading_score = (1.0 - cos_delta.clamp(-1, 1)).sum(dim=-1)                # (B*N,)

    # ── 3. 速度连续性：起步帧速度 vs 真实当前速度 ────────────────────────────
    xy0 = x0_denorm[:, 0, 0, :2]   # (B*N, 2): t=0 帧位置
    xy1 = ego_traj[:, 0, :2]        # (B*N, 2): t=1 帧位置
    speed_t1 = torch.norm(xy1 - xy0, dim=-1) / dt          # (B*N,) 估算速度(m/s)

    if ego_speed_gt is not None:
        # 与真实当前速度对比（更精确）
        spd_gt = ego_speed_gt.repeat_interleave(N)          # (B*N,)
        speed_score = (speed_t1 - spd_gt).pow(2)            # (B*N,)
    else:
        # 回退：速度应从 0 平滑启动（坐标系原点为自车）
        speed_score = speed_t1.pow(2)                        # (B*N,)

    # ── 综合得分（越小越优，先 minmax 归一化到 [0,1]） ───────────────────────
    def _minmax_norm(x):
        x = x.reshape(B, N)
        xmin = x.min(dim=1, keepdim=True).values
        xmax = x.max(dim=1, keepdim=True).values
        denom = (xmax - xmin).clamp(min=1e-6)
        return (x - xmin) / denom   # (B, N)

    score = (
        w_jerk    * _minmax_norm(jerk_score)
      + w_heading * _minmax_norm(heading_score)
      + w_speed   * _minmax_norm(speed_score)
    )  # (B, N)

    return score.argmin(dim=1)   # (B,)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._guidance_fn = config.guidance_fn

        # 推理时 ego anchor 初始化路径
        self._anchor_npz_path: str = getattr(
            config, "anchor_npz_path",
            "/home/xzl/diffusion_planner_test/vis_output/anchors.npz"
        )
        # truncated diffusion 参数
        self._num_anchors: int = getattr(config, "num_anchors", 3)
        self._t_start: float  = getattr(config, "t_start", 0.5)

    @property
    def sde(self):
        return self._sde

    def forward(self, encoder_outputs, inputs):
        ego_current = inputs['ego_current_state'][:, None, :4]          # (B, 1, 4)
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # (B, P, 4)
        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1)
            diffusion_time = inputs['diffusion_time']

            return {
                "score": self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask,
                ).reshape(B, P, -1, 4)
            }

        else:
            # ── 推理：随机选 N 条 anchor + Truncated Diffusion ────────────────
            N = self._num_anchors
            t_start = self._t_start

            # ── 预计算 current_states 的归一化版本，用于 t=0 帧约束 ──────────
            norm_mean = self._state_normalizer.mean.to(current_states.device)  # (P, 1, 4)
            norm_std  = self._state_normalizer.std.to(current_states.device)   # (P, 1, 4)
            current_states_norm = (current_states - norm_mean[:, 0, :].unsqueeze(0)) / norm_std[:, 0, :].unsqueeze(0)

            # ── 尝试获取真实速度（与训练 loss.py 的 _select_anchors_by_speed_v2 对齐） ──
            ego_speed = None
            ego_state_full = inputs.get("ego_current_state", None)
            if ego_state_full is not None and ego_state_full.shape[-1] >= 6:
                vx = ego_state_full[:, 4]
                vy = ego_state_full[:, 5]
                ego_speed = torch.sqrt(vx ** 2 + vy ** 2)  # (B,)

            # 1. 按速度桶选 N 条 anchor，批量插值归一化；ego_xT_multi: (B, N, 1+T, 4)
            ego_xT_multi = _build_ego_xT_multi(
                ego_current=current_states[:, 0, :],
                future_len=self._future_len,
                state_normalizer=self._state_normalizer,
                anchor_npz_path=self._anchor_npz_path,
                num_anchors=N,
                ego_speed=ego_speed,   # ← 与训练对齐：按速度桶选
            )

            # 2. 只对 future 帧（[1:]）加噪到 t_start（t=0 帧不加噪，更干净）
            ego_future_norm = ego_xT_multi[:, :, 1:, :]        # (B, N, T, 4)
            ego_future_flat = ego_future_norm.reshape(B * N, self._future_len, 4)
            t_tensor = torch.full((B * N,), t_start, device=ego_current.device, dtype=ego_current.dtype)
            mean_coeff, std_coeff = self._sde.marginal_prob(ego_future_flat, t_tensor)
            ego_future_noised = mean_coeff + std_coeff * torch.randn_like(ego_future_flat)  # (B*N, T, 4)

            # 拼回 t=0 帧（不加噪，直接用归一化当前状态）
            ego_t0_flat = ego_xT_multi[:, :, :1, :].reshape(B * N, 1, 4)  # (B*N, 1, 4)
            ego_xT_noised = torch.cat([ego_t0_flat, ego_future_noised], dim=1)  # (B*N, 1+T, 4)

            # 3. ⚠️ neighbors 未来帧置零（与训练 xT[:, 1:, 1:, :] = 0.0 对齐）
            neighbors_xT = torch.zeros(
                B * N, self._predicted_neighbor_num, 1 + self._future_len, 4,
                device=ego_current.device, dtype=ego_current.dtype
            )
            # neighbors t=0 帧填充归一化的当前状态（与训练 all_gt[:, :, :1, :] 对齐）
            current_states_norm_rep = current_states_norm.repeat_interleave(N, dim=0)  # (B*N, P, 4)
            neighbors_xT[:, :, 0, :] = current_states_norm_rep[:, 1:, :]  # t=0 帧：真实当前状态

            xT = torch.cat(
                [ego_xT_noised.unsqueeze(1), neighbors_xT], dim=1
            ).reshape(B * N, P, -1)  # (B*N, P, (1+T)*4)

            # 4. 扩展 condition 到 B*N
            ego_neighbor_encoding_rep = ego_neighbor_encoding.repeat_interleave(N, dim=0)
            route_lanes_rep           = route_lanes.repeat_interleave(N, dim=0)
            neighbor_mask_rep         = neighbor_current_mask.repeat_interleave(N, dim=0)

            # 5. 每步约束：锁定 t=0 帧，neighbors 未来帧继续置零（训练对齐）
            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B * N, P, -1, 4)
                # 锁定 t=0 帧
                xt[:, :, 0, :] = current_states_norm_rep
                # neighbors 未来帧置零（与训练对齐）
                xt[:, 1:, 1:, :] = 0.0
                return xt.reshape(B * N, P, -1)

            # 6. Truncated Diffusion：从 t_start 开始去噪
            guidance_type = "classifier" if self._guidance_fn is not None else "uncond"
            model_wrapper_params = {
                "guidance_type": guidance_type,
            }
            if self._guidance_fn is not None:
                model_wrapper_params.update({
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding_rep,
                            "route_lanes": route_lanes_rep,
                            "neighbor_current_mask": neighbor_mask_rep,
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer,
                    },
                    "guidance_scale": 0.5,
                })

            x0_raw = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding_rep,
                    "route_lanes": route_lanes_rep,
                    "neighbor_current_mask": neighbor_mask_rep,
                },
                dpm_solver_params={
                    "correcting_xt_fn": initial_state_constraint,
                },
                model_wrapper_params=model_wrapper_params,
                t_start=t_start,
            )
            # x0_raw: (B*N, P, (1+T)*4)

            # 7. 反归一化
            x0_denorm = self._state_normalizer.inverse(
                x0_raw.reshape(B * N, P, -1, 4)
            )  # (B*N, P, 1+T, 4)

            # 8. 综合候选选择：jerk 平滑度 + 方向一致性 + 速度连续性（传入真实速度）
            best_idx = _select_best_candidate(
                x0_denorm, B=B, N=N,
                w_jerk=1.0, w_heading=0.5, w_speed=0.3,
                ego_speed_gt=ego_speed,   # ← 利用真实速度改善精度
            )  # (B,)

            # 取最优候选
            x0 = x0_denorm.reshape(B, N, P, 1 + self._future_len, 4)
            best_idx_exp = best_idx[:, None, None, None, None].expand(
                B, 1, P, 1 + self._future_len, 4
            )
            x0_best = x0.gather(1, best_idx_exp).squeeze(1)   # (B, P, 1+T, 4)
            x0_best = x0_best[:, :, 1:]                        # 去掉 t=0 帧 → (B, P, T, 4)

            return {"prediction": x0_best}


class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        x = x[..., :4]
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)
        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)
        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))
        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()

        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        B, P, _ = x.shape

        x = self.preproj(x)

        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        x = x + x_embedding

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding + self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
