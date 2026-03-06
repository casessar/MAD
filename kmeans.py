import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.simulation.observation.observation_type import CameraChannel
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
DATA_ROOT   = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini/data/cache'
MAP_ROOT    = '/home/xzl/nuplan_dataset/nuplan-maps-v1.0/maps'
SENSOR_ROOT = '/home/xzl/nuplan_dataset/nuplan-v1.1_mini_camera_0/nuplan-v1.1_mini_camera_0'
# 使用所有 mini db 文件，保证场景多样性
# 只保留有相机数据的 log 对应的 db 文件
_cam_logs = {os.path.basename(p)
             for p in glob.glob(SENSOR_ROOT + '/*') if os.path.isdir(p)}
DB_FILES = sorted([
    p for p in glob.glob(DATA_ROOT + '/mini/*.db')
    if os.path.splitext(os.path.basename(p))[0] in _cam_logs
])
print(f"有相机数据的 db 文件数: {len(DB_FILES)}")
MAP_VERSION = 'nuplan-maps-v1.0'
OUT_DIR     = '/home/xzl/diffusion_planner_test/vis_output'
os.makedirs(OUT_DIR, exist_ok=True)
# 清空旧输出
for f in glob.glob(OUT_DIR + '/*.png'):
    os.remove(f)

# ─────────────────────────────────────────────
# 坐标变换：世界坐标 → 自车局部坐标
# ─────────────────────────────────────────────
def world_to_ego(wx, wy, ego_x, ego_y, ego_heading):
    """world (x,y) → ego local (前方=lx, 左方=ly)"""
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    dx = np.asarray(wx) - ego_x
    dy = np.asarray(wy) - ego_y
    lx =  cos_h * dx - sin_h * dy
    ly =  sin_h * dx + cos_h * dy
    return lx, ly

# ─────────────────────────────────────────────
# 坐标变换：自车局部坐标 → BEV 像素
# ─────────────────────────────────────────────
def ego_to_bev(lx, ly, cx, cy, px_per_m):
    """ego local (lx=前, ly=左) → BEV pixel (col, row)"""
    col = cx - np.asarray(ly) * px_per_m
    row = cy - np.asarray(lx) * px_per_m
    return col, row

# ─────────────────────────────────────────────
# 轨迹投影到前视相机图像
# ─────────────────────────────────────────────
def project_ego_pts_to_image(pts_ego_xy, cam_trans_inv, cam_intrinsic, img_w, img_h):
    """
    pts_ego_xy : (N,2) [前方x, 左方y]，z=0（地面）
    返回 : uv (M,2)，对应的原始点索引
    """
    N = len(pts_ego_xy)
    pts_h = np.ones((N, 4))
    pts_h[:, 0] = pts_ego_xy[:, 0]
    pts_h[:, 1] = pts_ego_xy[:, 1]
    pts_h[:, 2] = 0.0
    pts_cam = (cam_trans_inv @ pts_h.T)[:3, :]          # (3,N)
    valid   = pts_cam[2, :] > 0.1
    pts_v   = pts_cam[:, valid]
    idx_v   = np.where(valid)[0]
    if pts_v.shape[1] == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    uvw = cam_intrinsic @ pts_v
    uv  = (uvw[:2] / uvw[2:]).T                         # (M,2)
    in_img = ((uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
              (uv[:, 1] >= 0) & (uv[:, 1] < img_h))
    return uv[in_img], idx_v[in_img]

# ─────────────────────────────────────────────
# 裁剪 BEV Drivable Area 底图
# ─────────────────────────────────────────────
def crop_bev_map(raster_layer, ego_x, ego_y, ego_heading, half_range=50.0, px_per_m=4.0):
    T = raster_layer.transform
    raster_m_per_px = raster_layer.precision * abs(T[0, 0])   # 1.0 m/px
    col_ego = (ego_x - T[0, 3]) / T[0, 0]
    row_ego = (ego_y - T[1, 3]) / T[1, 1]
    crop_r  = int(half_range / raster_m_per_px)
    data    = raster_layer.data
    r0 = int(row_ego) - crop_r;  r1 = int(row_ego) + crop_r
    c0 = int(col_ego) - crop_r;  c1 = int(col_ego) + crop_r
    r0c = max(r0, 0); r1c = min(r1, data.shape[0])
    c0c = max(c0, 0); c1c = min(c1, data.shape[1])
    if r1c <= r0c or c1c <= c0c:
        out_px = int(half_range * 2 * px_per_m)
        return np.zeros((out_px, out_px), dtype=np.uint8)
    patch = data[r0c:r1c, c0c:c1c].copy()
    pad_top    = r0c - r0;  pad_bottom = r1 - r1c
    pad_left   = c0c - c0;  pad_right  = c1 - c1c
    if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode='constant', constant_values=0)
    from PIL import Image as PILImage
    angle_deg = 90.0 - np.degrees(ego_heading)
    img_rot   = PILImage.fromarray(patch).rotate(angle_deg, expand=False, fillcolor=0)
    out_px    = int(half_range * 2 * px_per_m)
    return np.array(img_rot.resize((out_px, out_px), resample=PILImage.BILINEAR))

# ─────────────────────────────────────────────
# 弧长等距插值函数
# ─────────────────────────────────────────────
def interp_traj(pts, step_m=0.5):
    """
    对轨迹点按弧长做等距插值，使相邻点间距不超过 step_m 米。
    pts: (N,2) numpy array
    返回: (M,2) numpy array
    """
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts)):
        seg_len = np.linalg.norm(pts[i] - pts[i-1])
        if seg_len < 1e-6:
            continue
        n_insert = int(np.ceil(seg_len / step_m))
        for k in range(1, n_insert + 1):
            out.append(pts[i-1] + (pts[i] - pts[i-1]) * k / n_insert)
    return np.array(out)

# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def visualize_scenarios(n_scenarios=20):
    HALF_RANGE  = 50.0    # BEV 显示范围（米）
    PX_PER_M    = 4.0     # BEV 像素/米
    TIME_HORIZON = 8.0    # 未来轨迹时长（秒）
    NUM_SAMPLES  = 16     # 未来轨迹采样点数

    # ── 1. 加载场景 ──────────────────────────────────────────────
    builder = NuPlanScenarioBuilder(
        data_root=DATA_ROOT, map_root=MAP_ROOT, sensor_root=SENSOR_ROOT,
        db_files=DB_FILES, map_version=MAP_VERSION, include_cameras=True,
    )
    sf = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=n_scenarios,
        expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
        timestamp_threshold_s=None, ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None, ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    scenarios = builder.get_scenarios(sf, Sequential())
    print(f"共加载 {len(scenarios)} 个场景")

    # ── 2. 读取相机内外参（用第一个 db 文件，标定参数所有 db 一致）────
    maps_db = GPKGMapsDB(MAP_VERSION, MAP_ROOT)
    db = NuPlanDB(data_root=DATA_ROOT, load_path=DB_FILES[0], maps_db=maps_db, verbose=False)
    cam_params = {cam.channel: {'intrinsic': cam.intrinsic_np,
                                'trans_inv': cam.trans_matrix_inv}
                  for cam in db.camera}
    CAM_CH   = CameraChannel.CAM_F0
    cam_info = cam_params.get('CAM_F0')
    if cam_info is None:
        raise RuntimeError("未找到 CAM_F0 标定参数")

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{n_scenarios}] {scenario.scenario_name}  |  {scenario.scenario_type}")

        # ── 3. 自车状态 ─────────────────────────────────────────
        ego      = scenario.get_ego_state_at_iteration(0)
        ego_x    = ego.center.x
        ego_y    = ego.center.y
        ego_head = ego.center.heading

        # 读取自车真实尺寸
        vp              = scenario.ego_vehicle_parameters
        ego_half_l      = vp.half_length           # 2.588 m，从几何中心量
        ego_half_w      = vp.half_width            # 1.1485 m
        rear_to_center  = vp.rear_axle_to_center   # 1.461 m，后轴→几何中心

        # ── 4. 自车未来轨迹（世界→ego局部）────────────────────
        # nuPlan ego.center 是后轴中心，轨迹点也是后轴中心
        # 为统一显示，全程以后轴为坐标原点，BEV 车框单独用 rear_to_center 偏移
        future_ego = scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=TIME_HORIZON, num_samples=NUM_SAMPLES)
        # 在最前面插入当前帧后轴位置，使轨迹从后轴出发（与车框中心一致）
        ego_pts = np.array([[ego_x, ego_y]] +
                           [[st.waypoint.x, st.waypoint.y] for st in future_ego])
        ego_pts = interp_traj(ego_pts)  # 插值轨迹点
        ego_lx, ego_ly = world_to_ego(ego_pts[:, 0], ego_pts[:, 1], ego_x, ego_y, ego_head)
        ego_local = np.stack([ego_lx, ego_ly], axis=1)   # (N,2)，第0点=(0,0)即后轴

        # ── 5. Agent 当前位置 + 未来轨迹 ────────────────────────
        cur_tracks = scenario.get_tracked_objects_at_iteration(0)
        cur_agents = cur_tracks.tracked_objects.get_agents()

        future_frames = list(scenario.get_future_tracked_objects(
            iteration=0, time_horizon=TIME_HORIZON, num_samples=NUM_SAMPLES))
        # 以当前帧位置作为轨迹第0个点
        agent_future: dict[str, list] = {
            a.track_token: [(a.center.x, a.center.y)] for a in cur_agents
        }
        for frame in future_frames:
            for ag in frame.tracked_objects.get_agents():
                if ag.track_token in agent_future:
                    agent_future[ag.track_token].append((ag.center.x, ag.center.y))

        # ── 6. 地图矢量元素（80m 半径内） ────────────────────────
        map_api   = scenario.map_api
        ego_pt    = Point2D(ego_x, ego_y)
        map_layers = [SemanticMapLayer.LANE,
                      SemanticMapLayer.LANE_CONNECTOR,
                      SemanticMapLayer.STOP_LINE,
                      SemanticMapLayer.CROSSWALK,
                      SemanticMapLayer.INTERSECTION]
        map_objs  = map_api.get_proximal_map_objects(ego_pt, HALF_RANGE * 1.2, map_layers)

        lanes          = map_objs.get(SemanticMapLayer.LANE, [])
        lane_connectors = map_objs.get(SemanticMapLayer.LANE_CONNECTOR, [])
        stop_lines     = map_objs.get(SemanticMapLayer.STOP_LINE, [])
        crosswalks     = map_objs.get(SemanticMapLayer.CROSSWALK, [])
        intersections  = map_objs.get(SemanticMapLayer.INTERSECTION, [])

        # ── 7. 前视相机图像 ──────────────────────────────────────
        sensors = scenario.get_sensors_at_iteration(iteration=0, channels=[CAM_CH])
        has_cam = sensors.images is not None and CAM_CH in sensors.images
        if has_cam:
            try:
                pil_img = sensors.images[CAM_CH].as_pil
                img_arr = np.array(pil_img)
                img_w, img_h = pil_img.size
                # 只投影未来轨迹点（ego_local[1:]），不含当前帧(0,0)
                # 当前帧后轴在相机后方，投影无意义
                uv_pts, _ = project_ego_pts_to_image(
                    ego_local[1:], cam_info['trans_inv'], cam_info['intrinsic'], img_w, img_h)
            except Exception as e:
                print(f"  ❌ 相机图像加载失败: {e}")
                has_cam = False

        # ── 8. BEV 底图（Drivable Area raster）──────────────────
        raster  = map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA)
        bev_raw = crop_bev_map(raster, ego_x, ego_y, ego_head,
                               half_range=HALF_RANGE, px_per_m=PX_PER_M)
        bev_h, bev_w = bev_raw.shape
        cx, cy = bev_w // 2, bev_h // 2

        # ── 9. 绘图 ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(26, 8))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#161b22')

        # -------- 子图1：相机原图 --------------------------------
        if has_cam:
            axes[0].imshow(img_arr)
        axes[0].set_title('Front Camera  (CAM_F0)', color='white', fontsize=12, pad=8)
        axes[0].axis('off')

        # -------- 子图2：轨迹投影到相机 -------------------------
        if has_cam:
            axes[1].imshow(img_arr)
            if len(uv_pts) > 0:
                # uv_pts 全是未来轨迹点，用 plasma 渐变色表示时间
                t_vals = np.linspace(0, 1, len(uv_pts))
                axes[1].scatter(uv_pts[:, 0], uv_pts[:, 1],
                                c=t_vals, cmap='plasma', s=50, zorder=5,
                                edgecolors='white', linewidths=0.5)
                axes[1].plot(uv_pts[:, 0], uv_pts[:, 1],
                             '-', color='cyan', lw=1.5, alpha=0.8, zorder=4)
            else:
                axes[1].text(img_w // 2, img_h // 2,
                             'No visible trajectory points',
                             color='yellow', ha='center', va='center', fontsize=11)
        axes[1].set_title('Ego Future Trajectory → Camera Projection',
                          color='white', fontsize=12, pad=8)
        axes[1].axis('off')
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, TIME_HORIZON))
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)
        cbar.set_label('Time (s)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        # -------- 子图3：BEV 全要素图 ---------------------------
        ax = axes[2]

        # 3a. Drivable Area 底图
        bev_rgb = np.zeros((*bev_raw.shape, 3), dtype=np.uint8)
        bev_rgb[bev_raw > 128] = [40, 55, 71]      # 可行驶：深蓝灰
        bev_rgb[bev_raw <= 128] = [13, 17, 23]     # 不可行驶：近黑
        ax.imshow(bev_rgb, origin='upper', zorder=0)

        # 3b. 辅助函数：polyline 的世界坐标 → BEV像素
        def poly_to_bev(pts_world):
            """pts_world: list of StateSE2 or Point2D (有.x .y 属性)"""
            wx = np.array([p.x for p in pts_world])
            wy = np.array([p.y for p in pts_world])
            lx, ly = world_to_ego(wx, wy, ego_x, ego_y, ego_head)
            # 过滤范围外的点
            mask = (np.abs(lx) < HALF_RANGE * 1.1) & (np.abs(ly) < HALF_RANGE * 1.1)
            col, row = ego_to_bev(lx, ly, cx, cy, PX_PER_M)
            return col[mask], row[mask]

        def polygon_to_bev(polygon):
            """shapely polygon → BEV像素坐标"""
            coords = np.array(polygon.exterior.coords)
            lx, ly = world_to_ego(coords[:, 0], coords[:, 1], ego_x, ego_y, ego_head)
            col, row = ego_to_bev(lx, ly, cx, cy, PX_PER_M)
            return col, row

        # 3c. Intersection 多边形（浅色填充）
        for intsec in intersections:
            try:
                col, row = polygon_to_bev(intsec.polygon)
                ax.fill(col, row, color='#2c3e50', alpha=0.6, zorder=1)
            except Exception:
                pass

        # 3d. Crosswalk 多边形（斑马线区域）
        for cw in crosswalks:
            try:
                col, row = polygon_to_bev(cw.polygon)
                ax.fill(col, row, color='#f39c12', alpha=0.25, zorder=2)
                ax.plot(np.append(col, col[0]), np.append(row, row[0]),
                        color='#f39c12', lw=0.8, alpha=0.7, zorder=2)
            except Exception:
                pass

        # 3e. 车道边界线（左右边界）和车道中心线
        for ln in lanes:
            try:
                # 左边界
                col, row = poly_to_bev(ln.left_boundary.discrete_path)
                if len(col) > 1:
                    ax.plot(col, row, color='#566573', lw=0.8, alpha=0.9, zorder=3)
                # 右边界
                col, row = poly_to_bev(ln.right_boundary.discrete_path)
                if len(col) > 1:
                    ax.plot(col, row, color='#566573', lw=0.8, alpha=0.9, zorder=3)
                # 中心线（虚线）
                col, row = poly_to_bev(ln.baseline_path.discrete_path)
                if len(col) > 1:
                    ax.plot(col, row, color='#f7f9f9', lw=0.6,
                            alpha=0.4, linestyle='--', dashes=(4, 6), zorder=3)
            except Exception:
                pass

        # 3f. Lane Connector（路口内虚线）
        for lc in lane_connectors:
            try:
                col, row = poly_to_bev(lc.baseline_path.discrete_path)
                if len(col) > 1:
                    ax.plot(col, row, color='#85929e', lw=0.8,
                            alpha=0.5, linestyle=':', zorder=3)
            except Exception:
                pass

        # 3g. Stop Line（红线）
        for sl in stop_lines:
            try:
                col, row = polygon_to_bev(sl.polygon)
                if len(col) > 1:
                    ax.fill(col, row, color='#e74c3c', alpha=0.5, zorder=4)
                    ax.plot(np.append(col, col[0]), np.append(row, row[0]),
                            color='#e74c3c', lw=1.2, zorder=4)
            except Exception:
                pass

        # 3h. Agent 当前位置（框）+ 未来轨迹
        for ag in cur_agents:
            ax_w = ag.box.half_width
            ax_l = ag.box.half_length
            ag_lx, ag_ly = world_to_ego(ag.center.x, ag.center.y, ego_x, ego_y, ego_head)
            if abs(ag_lx) > HALF_RANGE or abs(ag_ly) > HALF_RANGE:
                continue
            ag_col, ag_row = ego_to_bev(ag_lx, ag_ly, cx, cy, PX_PER_M)

            # 绘制 agent box（旋转矩形）
            rel_head = ag.center.heading - ego_head
            corners_local = np.array([
                [ ax_l,  ax_w], [ ax_l, -ax_w],
                [-ax_l, -ax_w], [-ax_l,  ax_w],
            ])
            rot = np.array([[np.cos(rel_head), -np.sin(rel_head)],
                            [np.sin(rel_head),  np.cos(rel_head)]])
            corners_rot = corners_local @ rot.T   # (4,2) in ego frame [front, left]
            box_col = ag_col - corners_rot[:, 1] * PX_PER_M
            box_row = ag_row - corners_rot[:, 0] * PX_PER_M
            ax.fill(np.append(box_col, box_col[0]),
                    np.append(box_row, box_row[0]),
                    color='#3498db', alpha=0.6, zorder=5)
            ax.plot(np.append(box_col, box_col[0]),
                    np.append(box_row, box_row[0]),
                    color='#5dade2', lw=0.8, zorder=5)

            # agent 未来轨迹
            traj = agent_future.get(ag.track_token, [])
            if len(traj) >= 2:
                wx_t = [p[0] for p in traj]
                wy_t = [p[1] for p in traj]
                lx_t, ly_t = world_to_ego(wx_t, wy_t, ego_x, ego_y, ego_head)
                col_t, row_t = ego_to_bev(lx_t, ly_t, cx, cy, PX_PER_M)
                # 颜色渐变线段
                pts_seg = np.array([col_t, row_t]).T.reshape(-1, 1, 2)
                segs    = np.concatenate([pts_seg[:-1], pts_seg[1:]], axis=1)
                t_norm  = np.linspace(0, 1, len(segs))
                lc_seg  = LineCollection(segs, cmap='Blues', norm=plt.Normalize(0, 1),
                                         linewidth=1.2, alpha=0.85, zorder=6)
                lc_seg.set_array(t_norm)
                ax.add_collection(lc_seg)

        # 3i. 自车未来轨迹（最显眼）
        ego_col, ego_row = ego_to_bev(ego_lx, ego_ly, cx, cy, PX_PER_M)
        t_ego = np.linspace(0, 1, len(ego_col))
        pts_seg = np.array([ego_col, ego_row]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([pts_seg[:-1], pts_seg[1:]], axis=1)
        lc_ego  = LineCollection(segs, cmap='plasma', norm=plt.Normalize(0, 1),
                                  linewidth=3.0, alpha=0.95, zorder=8)
        lc_ego.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc_ego)
        ax.scatter(ego_col, ego_row, c=t_ego, cmap='plasma',
                   s=18, zorder=9, edgecolors='none')

        # 3j. 自车矩形框
        # BEV 中 cx,cy = 后轴中心；几何中心在前方 rear_to_center 处
        el_px  = ego_half_l    * PX_PER_M   # 半车长（像素）
        ew_px  = ego_half_w    * PX_PER_M   # 半车宽（像素）
        r2c_px = rear_to_center * PX_PER_M  # 后轴→几何中心偏移（像素，BEV前方=row减小）
        # 几何中心在 BEV 中的像素位置
        geo_cx = cx
        geo_cy = cy - r2c_px   # 前方 → row 减小
        # 四角顺序：前左、前右、后右、后左
        box_cols = np.array([geo_cx - ew_px, geo_cx + ew_px,
                             geo_cx + ew_px, geo_cx - ew_px])
        box_rows = np.array([geo_cy - el_px, geo_cy - el_px,
                             geo_cy + el_px, geo_cy + el_px])
        ax.fill(np.append(box_cols, box_cols[0]),
                np.append(box_rows, box_rows[0]),
                color='#2ecc71', alpha=0.85, zorder=10)
        ax.plot(np.append(box_cols, box_cols[0]),
                np.append(box_rows, box_rows[0]),
                color='white', lw=1.2, zorder=10)
        # 后轴中心标记（与轨迹起点对齐）
        ax.plot(cx, cy, 'o', color='white', markersize=4, zorder=11)
        # 车头方向箭头（从几何中心指向前方车头外）
        ax.annotate('', xy=(geo_cx, geo_cy - el_px - 6), xytext=(geo_cx, geo_cy),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.8),
                    zorder=11)

        # 3k. 刻度和装饰
        ticks_m  = np.arange(-40, 41, 20)
        ticks_cx = [cx - t * PX_PER_M for t in ticks_m]
        ticks_cy = [cy - t * PX_PER_M for t in ticks_m]
        ax.set_xticks(ticks_cx)
        ax.set_xticklabels([f'{-int(t)}m' for t in ticks_m], color='#95a5a6', fontsize=7)
        ax.set_yticks(ticks_cy)
        ax.set_yticklabels([f'{int(t)}m' for t in ticks_m], color='#95a5a6', fontsize=7)
        ax.tick_params(colors='#95a5a6', length=2)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2c3e50')
        ax.set_xlim(0, bev_w); ax.set_ylim(bev_h, 0)

        # 3l. 图例
        legend_handles = [
            mpatches.Patch(color='#566573', label='Lane Boundary'),
            mpatches.Patch(color='#f7f9f9', alpha=0.4, label='Lane Centerline'),
            mpatches.Patch(color='#85929e', alpha=0.5, label='Lane Connector'),
            mpatches.Patch(color='#e74c3c', label='Stop Line'),
            mpatches.Patch(color='#f39c12', alpha=0.4, label='Crosswalk'),
            mpatches.Patch(color='#3498db', alpha=0.6, label='Agent (current)'),
            mpatches.Patch(color='#5dade2', alpha=0.8, label='Agent Future Traj'),
            mpatches.Patch(color='#e91e8c', label='Ego Future Traj'),
            mpatches.Patch(color='#2ecc71', label='Ego Vehicle'),
        ]
        ax.legend(handles=legend_handles, loc='lower left', fontsize=6.5,
                  facecolor='#0d1117', edgecolor='#2c3e50', labelcolor='white',
                  framealpha=0.85, ncol=1)
        ax.set_title('BEV: Vector Map + Agents + Ego Trajectory',
                     color='white', fontsize=12, pad=8)

        plt.suptitle(
            f"Scenario {i+1}:  {scenario.scenario_name}   |   {scenario.scenario_type}",
            color='white', fontsize=13, y=1.01,
        )
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'scenario_{i+1:02d}_{scenario.scenario_name}.png')
        plt.savefig(out_path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✅ 已保存: {out_path}")


if __name__ == '__main__':
    visualize_scenarios(n_scenarios=20)

"""
基于 nuPlan expert trajectory 的 K-Means Anchor 聚类（全场景版）

策略：
  - 静止场景（初始速度 < SPEED_STATIC）跳过，不参与聚类
  - 所有运动场景合并，做 K=15 的 K-Means
  - 额外手动追加 1 个全零静止 anchor（原点不动）
  - 最终 anchors.npz 包含：
      'moving'  shape=(15, T*2)   运动 anchor
      'static'  shape=(1,  T*2)   静止 anchor（全零）
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Optional

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

FUTURE_HORIZON  = 5.0    # 预测时长（秒）
SAMPLE_INTERVAL = 0.5    # 采样间隔（秒） → T=10 个点
NUM_CLUSTERS    = 15     # 运动 anchor 数量（静止另算）
SPEED_STATIC    = 0.5    # 低于此速度（m/s）视为静止，不参与聚类
N_TARGET        = 30000  # 目标收集样本数（够多即可，提前结束）
LOAD            = 50000  # 最多从 nuPlan 加载场景数

AXIS_XLIM = (-30, 30)
AXIS_YLIM = (-5,  85)


# ─────────────────────────────────────────────
# 单场景轨迹提取
# ─────────────────────────────────────────────
def extract_one(scenario) -> Optional[np.ndarray]:
    """
    返回归一化到自车坐标系的未来轨迹，shape=(T*2,)
    格式：[lx0, ly0, lx1, ly1, ...]  lx=前方, ly=左方
    初始速度 < SPEED_STATIC 时返回 None。
    """
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
            local.extend([
                cos_h * dx - sin_h * dy,   # lx：前方（纵向）
                sin_h * dx + cos_h * dy,   # ly：左方（横向）
            ])
        return np.array(local, dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────
# 可视化：单张图展示全部 15 条运动 anchor
# ─────────────────────────────────────────────
def visualize_anchors(moving_centers: np.ndarray, T: int,
                      n_samples: int, save_path: str):
    """
    将 K=15 条运动 anchor 画在同一张图中，用 tab20 区分颜色。
    静止 anchor（原点）用特殊标记标出。
    """
    K          = len(moving_centers)
    time_steps = np.arange(1, T + 1) * SAMPLE_INTERVAL  # [0.5, 1.0, ..., 5.0]

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

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor('white')

    # 背景
    ax.set_facecolor('#f7f9fc')
    ax.grid(True, color='#dce3ec', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b0bec5')
        spine.set_linewidth(0.8)

    # 零线
    ax.axhline(0, color='#90a4ae', lw=0.8, zorder=1)
    ax.axvline(0, color='#90a4ae', lw=0.8, zorder=1)

    # 用 tab20 给 15 条 anchor 分配颜色
    cmap_tab = matplotlib.colormaps.get_cmap('tab20')
    colors   = [cmap_tab(i / K) for i in range(K)]

    for k_idx, (row, color) in enumerate(zip(moving_centers, colors)):
        lx = row[0::2]   # 前方 → 纵轴
        ly = row[1::2]   # 左方 → 横轴

        # 从原点到第一个点的虚线
        ax.plot([0, ly[0]], [0, lx[0]],
                '--', color=color, lw=0.9, alpha=0.5, zorder=2)
        # 轨迹主线
        ax.plot(ly, lx, '-', color=color, lw=2.2, alpha=0.9, zorder=3,
                solid_capstyle='round', solid_joinstyle='round',
                label=f'Anchor {k_idx + 1:02d}')
        # 时间步散点（plasma 渐变表示时间进展）
        ax.scatter(ly, lx, c=time_steps, cmap='plasma',
                   vmin=0, vmax=FUTURE_HORIZON,
                   s=32, zorder=4, edgecolors='white', linewidths=0.4)
        # 终点
        ax.plot(ly[-1], lx[-1], 'o', color=color, markersize=7, zorder=5,
                markeredgecolor='white', markeredgewidth=0.9)

    # 静止 anchor 标记（原点处打叉）
    ax.plot(0, 0, 'x', color='#7f8c8d', markersize=10, markeredgewidth=2.2,
            zorder=9, label='Static anchor (origin)')

    # 自车三角形
    ax.plot(0, 0, marker=(3, 0, -90),
            color='#1a237e', markersize=18,
            markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label='Ego vehicle')

    # 标题与轴标签
    ax.set_title(
        f'K-Means Trajectory Anchors (All Scenarios)\n'
        f'K={K} moving  +  1 static  |  N={n_samples:,} samples',
        fontsize=13, fontweight='bold', color='#1a237e', pad=12,
    )
    ax.set_xlabel('Lateral displacement (m)', labelpad=7, color='#37474f')
    ax.set_ylabel('Longitudinal displacement (m)', labelpad=7, color='#37474f')
    ax.tick_params(colors='#546e7a', direction='in', length=3)

    # 统计信息框
    ax.text(0.98, 0.02,
            f'Horizon: {FUTURE_HORIZON:.0f}s @ {SAMPLE_INTERVAL}s\n'
            f'Total anchors: {K + 1}  (moving={K}, static=1)',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', color='#546e7a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#b0bec5', alpha=0.9))

    ax.set_xlim(AXIS_XLIM)
    ax.set_ylim(AXIS_YLIM)

    # 图例（放右上，最多两列避免遮挡）
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.88,
              edgecolor='#b0bec5', facecolor='white',
              ncol=2, handlelength=1.4, labelspacing=0.4)

    # 时间色条
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
    print(f"  可视化已保存 (300 dpi): {save_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    T = int(FUTURE_HORIZON / SAMPLE_INTERVAL)

    # ── 1. 加载场景 ──────────────────────────────────────────────
    print(f"加载 {len(DB_FILES)} 个 db，最多采样 {LOAD} 个场景（shuffle）...")
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

    # ── 2. 遍历场景，收集运动轨迹 ────────────────────────────────
    trajs   = []   # 所有运动场景的归一化轨迹
    n_static = 0   # 跳过的静止场景计数
    n_error  = 0   # 提取失败计数

    print(f"提取轨迹（目标 {N_TARGET} 条，速度 >= {SPEED_STATIC} m/s）...")
    for idx, sc in enumerate(scenarios):
        if len(trajs) >= N_TARGET:
            print(f"  已收集 {N_TARGET} 条，提前结束（处理了 {idx} 个场景）")
            break

        result = extract_one(sc)
        if result is None:
            # 区分静止与异常
            try:
                speed = sc.initial_ego_state.dynamic_car_state.speed
                if speed < SPEED_STATIC:
                    n_static += 1
                else:
                    n_error += 1
            except Exception:
                n_error += 1
            continue

        trajs.append(result)

        if idx % 500 == 0:
            print(f"  [{idx:>5d}/{len(scenarios)}]  已收集={len(trajs):,}"
                  f"  跳过静止={n_static}  异常={n_error}")

    print(f"\n===== 数据收集结果 =====")
    print(f"  运动轨迹: {len(trajs):,} 条")
    print(f"  跳过静止: {n_static}")
    print(f"  提取异常: {n_error}")

    if len(trajs) < NUM_CLUSTERS:
        raise RuntimeError(f"样本数 ({len(trajs)}) 不足 K={NUM_CLUSTERS}，请增大 LOAD。")

    # ── 3. K-Means 聚类（K=15，仅运动场景）────────────────────────
    data = np.array(trajs, dtype=np.float32)   # (N, T*2)
    print(f"\nK-Means  K={NUM_CLUSTERS}，样本={len(data)} ...")
    km = KMeans(n_clusters=NUM_CLUSTERS, random_state=42,
                n_init='auto', max_iter=500)
    km.fit(data)
    moving_centers = km.cluster_centers_   # (15, T*2)
    counts = np.bincount(km.labels_)
    print(f"  完成：簇大小 min={counts.min()}  max={counts.max()}"
          f"  mean={counts.mean():.1f}  inertia={km.inertia_:.2f}")

    # ── 4. 静止 anchor（全零，表示原地不动）─────────────────────
    static_anchor = np.zeros((1, T * 2), dtype=np.float32)   # (1, T*2)

    # ── 5. 保存 npz ──────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, 'anchors.npz')
    np.savez(npz_path, moving=moving_centers, static=static_anchor)
    print(f"\nAnchor 已保存: {npz_path}")
    print(f"  moving : shape={moving_centers.shape}  (K=15, T={T}, 每点2维)")
    print(f"  static : shape={static_anchor.shape}   (全零，原地不动)")
    print(f"  合计    : {NUM_CLUSTERS + 1} 个 anchor")

    # ── 6. 可视化 ────────────────────────────────────────────────
    vis_path = os.path.join(OUT_DIR, 'kmeans_anchors.png')
    visualize_anchors(moving_centers, T, len(trajs), vis_path)

    print(f"\n总耗时: {time.time()-t0:.1f}s")
"""
基于 nuPlan expert trajectory 的 K-Means Anchor 聚类
策略：
  - 过滤静止场景（speed < 0.5 m/s），其余所有运动场景统一聚类
  - K = 15（运动 anchor）+ 1 个手动追加的全零静止 anchor = 共 16 个
  - 保存字段：anchors['moving'] (15, T*2)，anchors['static'] (1, T*2)
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

FUTURE_HORIZON  = 5.0   # 预测时长（秒）
SAMPLE_INTERVAL = 0.5   # 采样间隔（秒）→ T = 10 个点
T               = int(FUTURE_HORIZON / SAMPLE_INTERVAL)

K_MOVING        = 15    # 运动场景聚类数
SPEED_STATIC    = 0.5   # 低于此速度视为静止，过滤掉（m/s）
LOAD            = 50000 # 最多加载场景数
N_TARGET        = 20000 # 目标收集运动轨迹数

# 可视化坐标轴范围
AXIS_XLIM = (-28, 28)
AXIS_YLIM = (-5,  80)


# ─────────────────────────────────────────────
# 单场景轨迹提取
# ─────────────────────────────────────────────
def extract_one(scenario):
    """
    提取单个场景的相对坐标轨迹。
    返回 np.ndarray (T*2,) 或 None（静止 / 数据不足 / 异常）。
    坐标系：以初始位置为原点，初始朝向为前方。
    layout: [lx_0, ly_0, lx_1, ly_1, ..., lx_{T-1}, ly_{T-1}]
      lx: 纵向（前方）位移
      ly: 横向（左方）位移
    """
    try:
        init  = scenario.initial_ego_state
        speed = init.dynamic_car_state.speed
        # 过滤静止场景
        if speed < SPEED_STATIC:
            return None

        ox, oy, oh = init.rear_axle.x, init.rear_axle.y, init.rear_axle.heading
        future = list(scenario.get_ego_future_trajectory(
            iteration=0, time_horizon=FUTURE_HORIZON, num_samples=T,
        ))
        if len(future) < T:
            return None

        cos_h, sin_h = np.cos(-oh)
        traj = []
        for st in future[:T]:
            dx = st.waypoint.x - ox
            dy = st.waypoint.y - oy
            lx =  cos_h * dx - sin_h * dy   # 纵向（前方）
            ly =  sin_h * dx + cos_h * dy   # 横向（左方）
            traj.extend([lx, ly])
        return np.array(traj, dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────
# 可视化：所有 moving anchor 画在一张图上
# ─────────────────────────────────────────────
def visualize_anchors(moving_centers: np.ndarray, n_samples: int, save_path: str):
    K          = len(moving_centers)
    time_steps = np.arange(1, T + 1) * SAMPLE_INTERVAL  # [0.5, 1.0, ..., 5.0]

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'axes.linewidth':    0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.patch.set_facecolor('white')

    ax.set_facecolor('#f7f9fc')
    ax.grid(True, color='#dce3ec', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#b0bec5')
        spine.set_linewidth(0.8)

    ax.axhline(0, color='#90a4ae', lw=0.8, zorder=1)
    ax.axvline(0, color='#90a4ae', lw=0.8, zorder=1)

    cmap = matplotlib.colormaps.get_cmap('tab20')

    for k_idx, row in enumerate(moving_centers):
        lx = row[0::2]   # 纵向 → 纵轴 Y
        ly = row[1::2]   # 横向 → 横轴 X
        color = cmap(k_idx / max(K - 1, 1))

        # 原点到第一个点的虚线
        ax.plot([0, ly[0]], [0, lx[0]],
                '--', color=color, lw=0.8, alpha=0.5, zorder=2)
        # 主轨迹线
        ax.plot(ly, lx, '-', color=color, lw=2.0, alpha=0.9, zorder=3,
                solid_capstyle='round', solid_joinstyle='round',
                label=f'Anchor {k_idx + 1:02d}')
        # 时间步散点（plasma 渐变表示时间）
        ax.scatter(ly, lx, c=time_steps, cmap='plasma',
                   vmin=0, vmax=FUTURE_HORIZON,
                   s=28, zorder=4, edgecolors='white', linewidths=0.4)
        # 终点标记
        ax.plot(ly[-1], lx[-1], 'o', color=color, markersize=6, zorder=5,
                markeredgecolor='white', markeredgewidth=0.8)

    # 静止 anchor 标记（原点圆圈）
    ax.plot(0, 0, 's', color='gray', markersize=9, zorder=9,
            markeredgecolor='white', markeredgewidth=1.2,
            label='Static anchor (zero)')

    # 自车标记
    ax.plot(0, 0, marker=(3, 0, -90),
            color='#1a6faf', markersize=16,
            markeredgecolor='white', markeredgewidth=1.5,
            zorder=10, label='Ego vehicle')

    ax.set_title(
        f'K-Means Trajectory Anchors (All Scenarios)\n'
        f'K = {K} moving  +  1 static  =  {K + 1} total',
        fontsize=13, fontweight='bold', color='#1a237e', pad=12,
    )
    ax.set_xlabel('Lateral displacement (m)', labelpad=7, color='#37474f')
    ax.set_ylabel('Longitudinal displacement (m)', labelpad=7, color='#37474f')
    ax.tick_params(colors='#546e7a', direction='in', length=3)

    ax.text(0.98, 0.02,
            f'N = {n_samples:,}  |  K_moving = {K}\n'
            f'Horizon: {FUTURE_HORIZON:.0f}s @ {SAMPLE_INTERVAL}s\n'
            f'Static threshold: v < {SPEED_STATIC} m/s',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', color='#546e7a',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#b0bec5', alpha=0.9))

    ax.set_xlim(AXIS_XLIM)
    ax.set_ylim(AXIS_YLIM)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.88,
              edgecolor='#b0bec5', facecolor='white', ncol=2)

    # 时间色条
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
    print(f"  可视化已保存 (300 dpi): {save_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import time
    t0 = time.time()

    # ── 1. 加载场景 ───────────────────────────────────────────────
    print(f"加载 {len(DB_FILES)} 个 db，最多采样 {LOAD} 个场景（shuffle）...")
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

    # ── 2. 提取所有运动场景轨迹 ───────────────────────────────────
    moving_trajs = []
    n_static_skipped = 0
    n_error_skipped  = 0

    print(f"提取轨迹（目标 {N_TARGET} 条运动轨迹）...")
    for idx, sc in enumerate(scenarios):
        if len(moving_trajs) >= N_TARGET:
            print(f"  已收集 {N_TARGET} 条，提前结束（处理了 {idx} 个场景）")
            break

        result = extract_one(sc)
        if result is None:
            # 粗略区分：速度过低 vs 其他异常（这里统一计入 static/error）
            n_static_skipped += 1
            continue

        moving_trajs.append(result)

        if idx % 500 == 0:
            print(f"  [{idx:>5d}/{len(scenarios)}]  运动轨迹: {len(moving_trajs):>6d}")

    print(f"\n===== 数据收集结果 =====")
    print(f"  运动场景轨迹:  {len(moving_trajs):>6d} 条")
    print(f"  跳过（静止/异常）: {n_static_skipped:>6d} 条")
    print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 3. K-Means 聚类（K=15） ───────────────────────────────────
    data = np.array(moving_trajs, dtype=np.float32)   # (N, T*2)
    k = min(K_MOVING, len(data))
    if k < K_MOVING:
        print(f"  ⚠ 样本不足，K: {K_MOVING} → {k}")

    print(f"\nK-Means 聚类  K={k}，样本={len(data)} ...")
    km = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
    km.fit(data)
    moving_centers = km.cluster_centers_   # (15, T*2)
    counts = np.bincount(km.labels_)
    print(f"  完成  簇大小: min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")
    print(f"  耗时: {time.time()-t0:.1f}s")

    # ── 4. 追加全零静止 anchor ────────────────────────────────────
    static_center = np.zeros((1, T * 2), dtype=np.float32)   # (1, T*2)
    all_centers   = np.concatenate([moving_centers, static_center], axis=0)  # (16, T*2)

    print(f"\n===== Anchor 汇总 =====")
    print(f"  moving anchors : shape={moving_centers.shape}   (K=15, T={T}, 每点2维)")
    print(f"  static anchor  : shape={static_center.shape}  (全零)")
    print(f"  总计           : {len(all_centers)} 个 anchor")

    # ── 5. 保存 npz ──────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, 'anchors.npz')
    np.savez(npz_path, moving=moving_centers, static=static_center)
    print(f"\nAnchor 已保存: {npz_path}")
    print(f"  载入方式: data = np.load('{npz_path}')")
    print(f"            data['moving'].shape  → {moving_centers.shape}")
    print(f"            data['static'].shape  → {static_center.shape}")

    # ── 6. 可视化 ────────────────────────────────────────────────
    vis_path = os.path.join(OUT_DIR, 'kmeans_anchors.png')
    visualize_anchors(moving_centers, n_samples=len(moving_trajs), save_path=vis_path)

    print(f"\n总耗时: {time.time()-t0:.1f}s")