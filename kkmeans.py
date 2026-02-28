"""
基于 nuPlan expert trajectory 的 K-Means Anchor 聚类
速度分桶（按 mini 数据集实际分布）：
  low : 0.5 ~ 5  m/s  (静止/低速)
  mid : 5  ~ 10 m/s  (中速)
  high: >= 10   m/s  (高速)
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Dict, Optional
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# ─────────────────────────────────────────────
# 路径 & 超参数配置
# ─────────────────────────────────────────────
DATA_ROOT    = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini/data/cache'
MAP_ROOT     = '/home/xzl/nuplan_dataset/nuplan-maps-v1.0/maps'
SENSOR_ROOT  = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini_camera_0/nuplan-v1.1_mini_camera_0'
DB_FILES     = sorted(glob.glob(DATA_ROOT + '/mini/*.db'))
MAP_VERSION  = 'nuplan-maps-v1.0'
OUT_DIR      = '/home/xzl/diffusion_planner_test/vis_output'
os.makedirs(OUT_DIR, exist_ok=True)

N_PER_BUCKET    = 10000        # 每个速度桶目标样本数
FUTURE_HORIZON  = 5.0          # 预测时长（秒）→ T=10 个点
SAMPLE_INTERVAL = 0.5          # 采样间隔（秒）
NUM_CLUSTERS    = {'low': 16, 'mid': 16, 'high': 16}

# 速度分桶边界（根据 mini 数据集实际分布调整）
# 静止场景（< 0.5 m/s）直接过滤，不参与聚类
# low: 0.5~5 m/s, mid: 5~10 m/s, high: >=10 m/s
SPEED_BINS  = {'low': (0.5, 5.0), 'mid': (5.0, 10.0), 'high': (10.0, 9999.0)}
SPEED_STATIC = 0.5   # 低于此速度视为静止，过滤掉

# 三张图统一坐标轴范围（以高速桶最大范围为基准，保证图大小一致）
AXIS_XLIM = (-28, 28)    # 横轴：左右方向
AXIS_YLIM = (-5,  80)    # 纵轴：前方方向


# ─────────────────────────────────────────────
# 单场景轨迹提取
# ─────────────────────────────────────────────
def extract_one(scenario) -> Optional[dict]:
    try:
        init  = scenario.initial_ego_state
        speed = init.dynamic_car_state.speed
        if speed < SPEED_STATIC:
            return None
        ox, oy, oh = init.rear_axle.x, init.rear_axle.y, init.rear_axle.heading

        T = int(FUTURE_HORIZON / SAMPLE_INTERVAL)
        future = list(scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=FUTURE_HORIZON, num_samples=T,
        ))
        if len(future) < T:
            return None

        cos_h, sin_h = np.cos(-oh), np.sin(-oh)
        local = []
        for st in future[:T]:
            dx = st.waypoint.x - ox
            dy = st.waypoint.y - oy
            local.extend([cos_h*dx - sin_h*dy,  # lx 前方
                          sin_h*dx + cos_h*dy])  # ly 左方
        return {'speed': speed, 'traj': local}
    except Exception:
        return None


def speed_bucket(speed: float) -> Optional[str]:
    for name, (lo, hi) in SPEED_BINS.items():
        if lo <= speed < hi:
            return name
    return None


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
def visualize_one_bucket(key: str, anchor_centers: np.ndarray, T: int,
                         n_samples: int, save_path: str):
    """论文级单桶可视化，独立保存为一张 png。"""

    BUCKET_CFG = {
        'low':  {'title': 'Low Speed Trajectory Anchors\n(0.5 – 5 m/s)',
                 'cmap': 'Blues',  'ego_color': '#1a6faf'},
        'mid':  {'title': 'Medium Speed Trajectory Anchors\n(5 – 10 m/s)',
                 'cmap': 'Greens', 'ego_color': '#1e8449'},
        'high': {'title': 'High Speed Trajectory Anchors\n(≥ 10 m/s)',
                 'cmap': 'Reds',   'ego_color': '#922b21'},
    }
    cfg        = BUCKET_CFG[key]
    time_steps = np.arange(1, T + 1) * SAMPLE_INTERVAL   # [0.5, 1.0, ..., 5.0]
    K          = len(anchor_centers)

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor('white')

    # ── 背景与网格 ────────────────────────────────────────────────
    ax.set_facecolor('#f7f9fc')
    ax.grid(True, color='#dce3ec', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b0bec5')
        spine.set_linewidth(0.8)   # 修复：方法调用而非赋值

    # ── 坐标轴零线 ────────────────────────────────────────────────
    ax.axhline(0, color='#90a4ae', lw=0.8, zorder=1)
    ax.axvline(0, color='#90a4ae', lw=0.8, zorder=1)

    # ── 绘制所有 anchor 轨迹 ──────────────────────────────────────
    cmap = matplotlib.colormaps.get_cmap(cfg['cmap'])
    for k_idx, row in enumerate(anchor_centers):
        lx = row[0::2]   # 前方 → 纵轴
        ly = row[1::2]   # 左方 → 横轴
        t_norm = 0.35 + 0.55 * (k_idx / max(K - 1, 1))
        color  = cmap(t_norm)

        # 从原点到第一个点的虚线
        ax.plot([0, ly[0]], [0, lx[0]],
                '--', color=color, lw=0.8, alpha=0.5, zorder=2)
        # 轨迹主线
        ax.plot(ly, lx, '-', color=color, lw=2.0, alpha=0.88, zorder=3,
                solid_capstyle='round', solid_joinstyle='round')
        # 时间步采样点（plasma 渐变色表示时间）
        ax.scatter(ly, lx, c=time_steps, cmap='plasma',
                   vmin=0, vmax=FUTURE_HORIZON,
                   s=28, zorder=4, edgecolors='white', linewidths=0.4)
        # 终点标记
        ax.plot(ly[-1], lx[-1], 'o', color=color, markersize=6, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)

    # ── 自车位置标记 ──────────────────────────────────────────────
    ax.plot(0, 0, marker=(3, 0, -90),
            color=cfg['ego_color'], markersize=16,
            markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label='Ego vehicle', clip_on=False)

    # ── 标题、轴标签 ──────────────────────────────────────────────
    ax.set_title(cfg['title'], fontsize=13, fontweight='bold',
                 color='#1a237e', pad=12)
    ax.set_xlabel('Lateral displacement (m)', labelpad=7, color='#37474f')
    ax.set_ylabel('Longitudinal displacement (m)', labelpad=7, color='#37474f')
    ax.tick_params(colors='#546e7a', direction='in', length=3)

    # ── 右下角统计信息框 ──────────────────────────────────────────
    ax.text(0.98, 0.02,
            f'N = {n_samples:,}  |  K = {K}\n'
            f'Horizon: {FUTURE_HORIZON:.0f}s @ {SAMPLE_INTERVAL}s',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', color='#546e7a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#b0bec5', alpha=0.9))

    ax.set_xlim(AXIS_XLIM)
    ax.set_ylim(AXIS_YLIM)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.88,
              edgecolor='#b0bec5', facecolor='white')

    # ── 时间色条 ──────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap='plasma', norm=plt.Normalize(vmin=0, vmax=FUTURE_HORIZON))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, aspect=28)
    cbar.set_label('Time (s)', fontsize=10, color='#37474f', labelpad=8)
    cbar.ax.tick_params(labelsize=9, colors='#546e7a')
    cbar.outline.set_edgecolor('#b0bec5')   # 修复：方法调用而非赋值

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  已保存 (300 dpi): {save_path}")


def visualize_anchors(anchors: Dict[str, np.ndarray], T: int,
                      buckets_raw: Dict[str, list], save_dir: str):
    """为每个速度桶单独保存一张论文图。"""
    names = {'low': 'anchor_low_speed', 'mid': 'anchor_mid_speed', 'high': 'anchor_high_speed'}
    for key in ['low', 'mid', 'high']:
        if key not in anchors:
            print(f"  ⚠ '{key}' 无 anchor，跳过。")
            continue
        save_path = os.path.join(save_dir, f'{names[key]}.png')
        visualize_one_bucket(
            key=key,
            anchor_centers=anchors[key],
            T=T,
            n_samples=len(buckets_raw.get(key, [])),
            save_path=save_path,
        )


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    T = int(FUTURE_HORIZON / SAMPLE_INTERVAL)

    # ── 1. 流式加载场景，按初始速度分桶，凑满即停 ───────────────
    # 一次加载大批场景（shuffle保证多样性），边遍历边分桶
    # mid 桶占比约 11%，high 桶占比约 10%（调整边界后）
    # 需要约 1000/0.10 = 10000 个场景才能凑满 high 桶
    # 多取一些保险，设 LOAD = 10000
    LOAD = 50000
    print(f"加载 {len(DB_FILES)} 个 db，采样 {LOAD} 个场景（shuffle）...")

    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=DB_FILES, map_version=MAP_VERSION, include_cameras=False,
    )
    sf = ScenarioFilter(
        scenario_types=None, scenario_tokens=None,
        log_names=None, map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=LOAD,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    scenarios = builder.get_scenarios(sf, Sequential())
    print(f"实际加载: {len(scenarios)} 个场景  ({time.time()-t0:.1f}s)\n")

    # ── 2. 遍历场景，按初始速度手动分桶 ─────────────────────────
    buckets: Dict[str, list] = {'low': [], 'mid': [], 'high': []}
    skipped = 0

    print(f"提取轨迹并分桶（目标每桶 {N_PER_BUCKET} 个样本）...")
    for idx, sc in enumerate(scenarios):
        # 若三个桶都已凑满，提前退出
        if all(len(v) >= N_PER_BUCKET for v in buckets.values()):
            print(f"  所有桶已满，提前结束（处理了 {idx} 个场景）")
            break

        res = extract_one(sc)
        if res is None:
            skipped += 1
            continue

        bkt = speed_bucket(res['speed'])
        if bkt is None:
            continue
        # 该桶已满则跳过，让其他桶继续收集
        if len(buckets[bkt]) >= N_PER_BUCKET:
            continue

        buckets[bkt].append(res['traj'])

        if idx % 200 == 0:
            status = '  '.join(f"{k}={len(v)}/{N_PER_BUCKET}" for k, v in buckets.items())
            print(f"  [{idx:>4d}/{len(scenarios)}]  {status}")

    print(f"\n===== 数据收集结果（跳过异常: {skipped}）=====")
    for key, data in buckets.items():
        lo, hi = SPEED_BINS[key]
        hi_str = f"{hi:.0f}" if hi < 9999 else "∞"
        print(f"  '{key}' ({lo:.0f}–{hi_str} m/s): {len(data):4d} 条样本")

    # ── 3. K-Means 聚类 ──────────────────────────────────────────
    anchors: Dict[str, np.ndarray] = {}
    for key, raw in buckets.items():
        data = np.array(raw, dtype=np.float32)
        k = min(NUM_CLUSTERS[key], len(data))
        if k < 2:
            print(f"  ⚠ '{key}' 样本过少（{len(data)}），跳过聚类。")
            continue
        if k < NUM_CLUSTERS[key]:
            print(f"  ⚠ '{key}' 样本不足，K: {NUM_CLUSTERS[key]} → {k}")
        print(f"\nK-Means '{key}'  K={k}，样本={len(data)} ...")
        km = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
        km.fit(data)
        anchors[key] = km.cluster_centers_   # (K, T*2)
        counts = np.bincount(km.labels_)
        print(f"  完成：簇大小 min={counts.min()}  max={counts.max()}"
              f"  mean={counts.mean():.1f}")

    # ── 4. 保存 npz ──────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, 'anchors.npz')
    np.savez(npz_path, **anchors)
    print(f"\nAnchor 矩阵已保存: {npz_path}")
    for key, val in anchors.items():
        print(f"  anchors['{key}']: shape={val.shape}  "
              f"(K={val.shape[0]}, T={T}, 每点2维)")

    # ── 5. 可视化 ────────────────────────────────────────────────
    visualize_anchors(anchors, T, buckets, OUT_DIR)

    print(f"\n总耗时: {time.time()-t0:.1f}s")