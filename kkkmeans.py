"""
基于 nuPlan expert trajectory 的 K-Means Anchor 聚类（全场景版）
- 不区分速度，过滤静止场景（< 0.5 m/s）后对所有场景聚 K=15 类
- 额外保留 1 个静止 anchor（全零轨迹），共 16 个 anchor
- 保存为 anchors_all.npz，包含 key: 'all'(15,20) 和 'static'(1,20)
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# ─────────────────────────────────────────────
# 路径 & 超参数配置
# ─────────────────────────────────────────────
DATA_ROOT   = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini/data/cache'
MAP_ROOT    = '/home/xzl/nuplan_dataset/nuplan-maps-v1.0/maps'
SENSOR_ROOT = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini_camera_0/nuplan-v1.1_mini_camera_0'
DB_FILES    = sorted(glob.glob(DATA_ROOT + '/mini/*.db'))
MAP_VERSION = 'nuplan-maps-v1.0'
OUT_DIR     = '/home/xzl/diffusion_planner_test/vis_output'
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES       = 30000   # 目标采样数量
FUTURE_HORIZON  = 5.0     # 预测时长（秒）→ T=10 个点
SAMPLE_INTERVAL = 0.5     # 采样间隔（秒）
NUM_CLUSTERS    = 15      # 运动场景聚类数（第16个留给静止）
SPEED_STATIC    = 0.5     # 低于此速度视为静止，过滤

AXIS_XLIM = (-28, 28)
AXIS_YLIM = (-5,  80)


# ─────────────────────────────────────────────
# 单场景轨迹提取
# ─────────────────────────────────────────────
def extract_one(scenario):
    try:
        init  = scenario.initial_ego_state
        speed = init.dynamic_car_state.speed
        if speed < SPEED_STATIC:
            return None   # 静止场景直接过滤
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
            local.extend([cos_h*dx - sin_h*dy,   # lx 前方
                          sin_h*dx + cos_h*dy])   # ly 左方
        return local
    except Exception:
        return None


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
def visualize_anchors_all(centers: np.ndarray, T: int, n_samples: int, save_path: str):
    """将全部 K=15 个运动 anchor 可视化为一张图。"""
    time_steps = np.arange(1, T + 1) * SAMPLE_INTERVAL
    K = len(centers)

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'axes.linewidth':    0.8,
    })

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f7f9fc')
    ax.grid(True, color='#dce3ec', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b0bec5')
        spine.set_linewidth(0.8)
    ax.axhline(0, color='#90a4ae', lw=0.8, zorder=1)
    ax.axvline(0, color='#90a4ae', lw=0.8, zorder=1)

    cmap = matplotlib.colormaps.get_cmap('viridis')
    for k_idx, row in enumerate(centers):
        lx = row[0::2]   # 前方 → 纵轴
        ly = row[1::2]   # 左方 → 横轴
        t_norm = 0.2 + 0.75 * (k_idx / max(K - 1, 1))
        color  = cmap(t_norm)

        ax.plot([0, ly[0]], [0, lx[0]],
                '--', color=color, lw=0.8, alpha=0.5, zorder=2)
        ax.plot(ly, lx, '-', color=color, lw=2.0, alpha=0.88, zorder=3,
                solid_capstyle='round', solid_joinstyle='round')
        ax.scatter(ly, lx, c=time_steps, cmap='plasma',
                   vmin=0, vmax=FUTURE_HORIZON,
                   s=28, zorder=4, edgecolors='white', linewidths=0.4)
        ax.plot(ly[-1], lx[-1], 'o', color=color, markersize=6, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)

    # 静止 anchor 单独标一下（原点）
    ax.plot(0, 0, 's', color='#e74c3c', markersize=8, zorder=9,
            markeredgecolor='white', markeredgewidth=1.2,
            label='Static anchor (v < 0.5 m/s)')

    # 自车位置
    ax.plot(0, 0, marker=(3, 0, -90), color='#1a6faf', markersize=16,
            markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label='Ego vehicle', clip_on=False)

    ax.set_title(f'All-scenario Trajectory Anchors (K={K}+1 static)',
                 fontsize=13, fontweight='bold', color='#1a237e', pad=12)
    ax.set_xlabel('Lateral displacement (m)', labelpad=7, color='#37474f')
    ax.set_ylabel('Longitudinal displacement (m)', labelpad=7, color='#37474f')
    ax.tick_params(colors='#546e7a', direction='in', length=3)

    ax.text(0.98, 0.02,
            f'N = {n_samples:,}  |  K = {K} + 1 static\n'
            f'Horizon: {FUTURE_HORIZON:.0f}s @ {SAMPLE_INTERVAL}s',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', color='#546e7a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#b0bec5', alpha=0.9))

    ax.set_xlim(AXIS_XLIM)
    ax.set_ylim(AXIS_YLIM)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.88,
              edgecolor='#b0bec5', facecolor='white')

    sm = plt.cm.ScalarMappable(
        cmap='plasma', norm=plt.Normalize(vmin=0, vmax=FUTURE_HORIZON))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, aspect=28)
    cbar.set_label('Time (s)', fontsize=10, color='#37474f', labelpad=8)
    cbar.ax.tick_params(labelsize=9, colors='#546e7a')
    cbar.outline.set_edgecolor('#b0bec5')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  已保存 (300 dpi): {save_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    T = int(FUTURE_HORIZON / SAMPLE_INTERVAL)

    # ── 1. 加载场景 ──────────────────────────────────────────────
    LOAD = 50000
    print(f"加载 {len(DB_FILES)} 个 db，采样最多 {LOAD} 个场景（shuffle）...")

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

    # ── 2. 遍历场景，提取运动轨迹（过滤静止） ───────────────────
    trajs = []
    skipped = 0
    print(f"提取轨迹（过滤 v < {SPEED_STATIC} m/s，目标 {N_SAMPLES} 条）...")
    for idx, sc in enumerate(scenarios):
        if len(trajs) >= N_SAMPLES:
            print(f"  已达目标 {N_SAMPLES} 条，提前结束（处理了 {idx} 个场景）")
            break
        res = extract_one(sc)
        if res is None:
            skipped += 1
            continue
        trajs.append(res)
        if idx % 500 == 0:
            print(f"  [{idx:>5d}/{len(scenarios)}]  收集: {len(trajs)}")

    print(f"\n===== 数据收集结果 =====")
    print(f"  运动场景样本: {len(trajs)} 条  |  跳过（静止/异常）: {skipped}")

    # ── 3. K-Means 聚类（K=15，不含静止） ───────────────────────
    data = np.array(trajs, dtype=np.float32)   # (N, 20)
    k = min(NUM_CLUSTERS, len(data))
    print(f"\nK-Means  K={k}，样本={len(data)} ...")
    km = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
    km.fit(data)
    centers = km.cluster_centers_   # (15, 20)
    counts  = np.bincount(km.labels_)
    print(f"  完成：簇大小 min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")

    # ── 4. 静止 anchor（全零轨迹） ───────────────────────────────
    static_anchor = np.zeros((1, T * 2), dtype=np.float32)  # (1, 20)
    print(f"\n静止 anchor: shape={static_anchor.shape}  (全零轨迹)")

    # ── 5. 保存 npz ──────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, 'anchors_all.npz')
    np.savez(npz_path, all=centers, static=static_anchor)
    print(f"\nAnchor 已保存: {npz_path}")
    print(f"  anchors['all']:    shape={centers.shape}")
    print(f"  anchors['static']: shape={static_anchor.shape}")

    # ── 6. 可视化 ────────────────────────────────────────────────
    vis_path = os.path.join(OUT_DIR, 'kmeans_anchors_all.png')
    visualize_anchors_all(centers, T, len(trajs), vis_path)

    print(f"\n总耗时: {time.time()-t0:.1f}s")