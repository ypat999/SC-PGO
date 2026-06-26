#!/usr/bin/env python3
"""
Super-LIO 离线回环检测工具 (BTC + GICP)

读取 odom_poses.txt（KITTI 格式）和 Scans/*.pcd 文件，
使用 BTC (Binary Triangle Context) 描述子检测回环，
使用 GICP 验证回环并执行位姿图优化。

输出文件：
  - optimized_poses.txt  : KITTI 格式优化后的轨迹
  - loop_pairs.txt       : 检测到的回环对 (frame_a frame_b btc_score fitness_score)
  - loop_details.txt     : 详细的回环信息

用法：
  python3 offline_loop_closure.py
  python3 offline_loop_closure.py /path/to/save_data/ --sc-dist-thres 0.25
  python3 offline_loop_closure.py --no-gicp --skip-optimization
"""

import os
import sys
import argparse
import numpy as np
from numpy import linalg as LA
import time
from collections import defaultdict

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("[WARN] open3d 未安装，GICP 验证和地图合并功能已禁用。")
    print("       安装: pip install open3d")

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    print("[WARN] gtsam (GTSAM Python) 未安装，位姿图优化功能已禁用。")
    print("       安装: pip install gtsam")

# ======================== 默认配置 ========================
DEFAULT_DATA_DIR = "/home/ywj/save_data/"

# ======================== BTC 参数 ========================
BTC_VOXEL_SIZE = 1.0                # 体素大小 (m)
BTC_PLANE_EIGEN_RATIO = 0.05       # 平面检测: 最小特征值/次小特征值阈值
BTC_PLANE_MIN_POINTS = 10           # 平面最小点数
BTC_DESC_GRID_SIZE = 10             # 二进制描述子网格大小 (N x N)
BTC_DESC_PROJ_RADIUS = 3.0          # 描述子投影半径 (m)
BTC_DESC_GRID_RES = BTC_DESC_PROJ_RADIUS * 2 / BTC_DESC_GRID_SIZE

BTC_TRIANGLE_MIN_SIDE = 1.0         # 三角形最小边长 (m)
BTC_TRIANGLE_MAX_SIDE = 10.0        # 三角形最大边长 (m)
BTC_MAX_DESCS_PER_FRAME = 100       # 每帧最多保留的描述子数

# BTC 回环检测参数
BTC_MATCH_SCORE_THRES = 0.3         # 描述子匹配分数阈值
BTC_NUM_CANDIDATES = 10             # 每帧搜索的候选帧数
NUM_EXCLUDE_RECENT = 30             # 排除最近的帧数

# GICP 验证参数
GICP_MAX_DIST = 30.0                # GICP 最大对应距离
GICP_ITERATIONS = 100               # GICP 最大迭代次数
GICP_FITNESS_THRES = 0.5            # GICP fitness score 阈值
GICP_EPSILON = 1e-6                 # 收敛阈值

# ======================== 工具函数 ========================

def load_poses(filepath):
    """加载 KITTI 格式位姿（每行 12 列: R3x3 + t3x1）。"""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) != 12:
                print(f"[WARN] 跳过无效位姿行: {line[:50]}...")
                continue
            R = np.array(vals[:9]).reshape(3, 3)
            t = np.array(vals[9:12]).reshape(3, 1)
            T = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
            poses.append(T)
    return poses


def save_poses_kitti(filepath, poses):
    """保存位姿为 KITTI 格式（每行 12 列）。"""
    with open(filepath, 'w') as f:
        for T in poses:
            R = T[:3, :3]
            t = T[:3, 3]
            f.write(f"{R[0,0]:.10f} {R[0,1]:.10f} {R[0,2]:.10f} {t[0]:.10f} "
                    f"{R[1,0]:.10f} {R[1,1]:.10f} {R[1,2]:.10f} {t[1]:.10f} "
                    f"{R[2,0]:.10f} {R[2,1]:.10f} {R[2,2]:.10f} {t[2]:.10f}\n")


def save_loop_pairs(filepath, pairs):
    """保存回环对: frame_a frame_b btc_score fitness_score。"""
    with open(filepath, 'w') as f:
        f.write("# frame_a frame_b btc_score fitness_score\n")
        for a, b, s, fscore in pairs:
            f.write(f"{a} {b} {s:.6f} {fscore:.6f}\n")


def load_scans_o3d(scan_dir, num_poses):
    """使用 Open3D 加载 PCD 点云，返回 (points_np, o3d_pcd) 列表。"""
    scans = []
    for i in range(num_poses):
        fname = f"{i:06d}.pcd"
        fpath = os.path.join(scan_dir, fname)
        if os.path.exists(fpath):
            pcd = o3d.io.read_point_cloud(fpath)
            pts = np.asarray(pcd.points)
            scans.append((pts, pcd))
        else:
            scans.append((np.empty((0, 3)), o3d.geometry.PointCloud()))
    return scans


# ======================== BTC 描述子实现 ========================

class BinaryDescriptor:
    """BTC 二进制描述子：表示一个平面块的局部特征"""
    __slots__ = ('x_2d', 'y_2d', 'value', 'pose_3d', 'normal', 'frame_id')

    def __init__(self, x_2d, y_2d, value, pose_3d, normal, frame_id):
        self.x_2d = x_2d        # 平面局部坐标系中的 2D x
        self.y_2d = y_2d        # 平面局部坐标系中的 2D y
        self.value = value      # 摘要哈希值
        self.pose_3d = pose_3d  # 3D 世界坐标
        self.normal = normal    # 法向量
        self.frame_id = frame_id


class BTC:
    """Binary Triangle Context 描述子：三个描述子构成的三角形"""
    __slots__ = ('triangle_sides', 'angles', 'center', 'frame_id', 'desc_a', 'desc_b', 'desc_c')

    def __init__(self, sides, angles, center, frame_id, da, db, dc):
        self.triangle_sides = sides    # 三角形边长 (3,)
        self.angles = angles           # 三角形角度 (3,)
        self.center = center           # 三角形中心 3D 坐标
        self.frame_id = frame_id
        self.desc_a = da
        self.desc_b = db
        self.desc_c = dc


def detect_planes_pca(points, voxel_size=BTC_VOXEL_SIZE,
                      eigen_ratio=BTC_PLANE_EIGEN_RATIO,
                      min_points=BTC_PLANE_MIN_POINTS):
    """
    使用体素化 + PCA 检测平面。
    返回 planes: [(center_3d, normal_3d, plane_points_3d, eigenvals), ...]
    """
    if len(points) < min_points:
        return []

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # 体素索引
    grid_size = np.maximum(max_bound - min_bound, 1e-6)
    voxel_shape = np.ceil(grid_size / voxel_size).astype(int) + 1
    voxel_dict = defaultdict(list)

    for i, pt in enumerate(points):
        idx = tuple(((pt - min_bound) / voxel_size).astype(int))
        voxel_dict[idx].append(i)

    planes = []
    for idx, pt_indices in voxel_dict.items():
        if len(pt_indices) < min_points:
            continue
        voxel_pts = points[pt_indices]

        # PCA 检测平面度
        center = voxel_pts.mean(axis=0)
        cov = np.cov((voxel_pts - center).T)
        if cov.shape != (3, 3):
            continue
        eigenvals, eigenvecs = LA.eigh(cov)
        # eigenvals 从小到大排序
        ratio = eigenvals[0] / (eigenvals[1] + 1e-12)
        if ratio < eigen_ratio:
            normal = eigenvecs[:, 0]
            # 确保法向量朝上
            if normal[2] < 0:
                normal = -normal
            planes.append((center, normal, voxel_pts, eigenvals))

    return planes


def build_binary_descriptor(center, normal, points_in_plane,
                            grid_size=BTC_DESC_GRID_SIZE,
                            proj_radius=BTC_DESC_PROJ_RADIUS):
    """
    从平面点构建二进制描述子。
    将平面点投影到平面局部 2D 坐标系，离散化为二值网格。
    返回 (grid, local_pts_2d, summary_value)。
    """
    # 构建局部坐标系（平面为 XY 平面，法向量为 Z）
    # 选择一个与法向量不平行的参考向量
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / (LA.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (LA.norm(v) + 1e-12)

    # 将平面点投影到局部 2D 坐标
    pts_centered = points_in_plane - center
    x_2d = np.dot(pts_centered, u)
    y_2d = np.dot(pts_centered, v)

    # 过滤掉超出投影半径的点
    mask = (np.abs(x_2d) < proj_radius) & (np.abs(y_2d) < proj_radius)
    x_2d = x_2d[mask]
    y_2d = y_2d[mask]

    if len(x_2d) < 3:
        return None, None, 0

    # 离散化为二值网格
    grid_res = 2 * proj_radius / grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for x, y in zip(x_2d, y_2d):
        gi = int((x + proj_radius) / grid_res)
        gj = int((y + proj_radius) / grid_res)
        gi = np.clip(gi, 0, grid_size - 1)
        gj = np.clip(gj, 0, grid_size - 1)
        grid[gi, gj] = 1

    # 计算摘要值（用于快速匹配）
    summary = int(grid.sum()) ^ hash(grid.tobytes()[:16]) & 0xFFFF

    # 取网格中心作为描述子的 2D 位置
    x_center = x_2d.mean()
    y_center = y_2d.mean()
    center_3d = center  # 3D 位置即平面中心

    return grid, (x_center, y_center, center_3d, normal), summary


def generate_btcs_for_frame(points, frame_id, poses=None):
    """
    为一帧点云生成 BTC 描述子。
    返回 [BTC, ...]
    """
    if len(points) < BTC_PLANE_MIN_POINTS:
        return []

    # 步骤 1: 体素化检测平面
    planes = detect_planes_pca(points)
    if len(planes) < 3:
        return []

    # 步骤 2: 为每个平面生成二进制描述子
    descriptors = []
    for center, normal, plane_pts, _ in planes:
        grid, local_info, summary = build_binary_descriptor(
            center, normal, plane_pts)
        if grid is not None and summary != 0:
            x_2d, y_2d, center_3d, norm = local_info
            desc = BinaryDescriptor(
                x_2d=float(x_2d),
                y_2d=float(y_2d),
                value=summary,
                pose_3d=center_3d,
                normal=norm,
                frame_id=frame_id
            )
            descriptors.append(desc)

    if len(descriptors) < 3:
        return []

    # 限制描述子数量（按 value 去重并选 top-N 个不同 value 的）
    uniq_descs = {}
    for d in descriptors:
        val = d.value
        if val not in uniq_descs:
            uniq_descs[val] = d
    descriptors = list(uniq_descs.values())
    if len(descriptors) > BTC_MAX_DESCS_PER_FRAME:
        descriptors = sorted(descriptors, key=lambda d: d.value)[:BTC_MAX_DESCS_PER_FRAME]

    # 步骤 3: 构建三角形 BTC
    btcs = []
    n = len(descriptors)
    # 使用空间哈希分组，只选择空间上相近的描述子构成三角形
    voxel_map = defaultdict(list)
    for i, d in enumerate(descriptors):
        vx = int(d.pose_3d[0] / BTC_TRIANGLE_MAX_SIDE)
        vy = int(d.pose_3d[1] / BTC_TRIANGLE_MAX_SIDE)
        vz = int(d.pose_3d[2] / BTC_TRIANGLE_MAX_SIDE)
        voxel_map[(vx, vy, vz)].append(i)

    # 从同一体素或相邻体素中选择 3 个描述子构成三角形
    visited_triplets = set()
    for voxel_key, indices in voxel_map.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                for k in range(j + 1, len(indices)):
                    di, dj, dk = indices[i], indices[j], indices[k]
                    triplet = tuple(sorted([di, dj, dk]))
                    if triplet in visited_triplets:
                        continue
                    visited_triplets.add(triplet)

                    da = descriptors[di]
                    db = descriptors[dj]
                    dc = descriptors[dk]

                    # 计算三角形边长
                    a = LA.norm(da.pose_3d - db.pose_3d)
                    b = LA.norm(db.pose_3d - dc.pose_3d)
                    c = LA.norm(dc.pose_3d - da.pose_3d)
                    sides = np.array([a, b, c])

                    # 过滤非法三角形
                    if (np.min(sides) < BTC_TRIANGLE_MIN_SIDE or
                        np.max(sides) > BTC_TRIANGLE_MAX_SIDE):
                        continue

                    # 三角形不等式检查
                    if a + b <= c or b + c <= a or c + a <= b:
                        continue

                    # 计算角度
                    angles = np.zeros(3)
                    angles[0] = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-12), -1, 1))
                    angles[1] = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c + 1e-12), -1, 1))
                    angles[2] = np.pi - angles[0] - angles[1]

                    # 三角形中心
                    center = (da.pose_3d + db.pose_3d + dc.pose_3d) / 3.0

                    btc = BTC(sides, angles, center, frame_id, da, db, dc)
                    btcs.append(btc)

                    if len(btcs) >= BTC_MAX_DESCS_PER_FRAME:
                        return btcs

    return btcs


def btc_hash_key(btc):
    """计算 BTC 的哈希键：体素位置 + 三角形形状编码"""
    # 量化为 2m 体素
    vx = int(btc.center[0] / 2.0)
    vy = int(btc.center[1] / 2.0)
    vz = int(btc.center[2] / 2.0)

    # 三角形形状编码：边长排序后量化为 0.5m 分辨率
    sorted_sides = sorted(btc.triangle_sides)
    side_code = tuple(int(s / 0.5) for s in sorted_sides)
    angle_code = tuple(int(a / 0.1) for a in sorted(btc.angles))

    return (vx, vy, vz, side_code, angle_code)


def compute_btc_similarity(btc1, btc2):
    """
    计算两个 BTC 的相似度 [0, 1]，1 表示完全相同。
    综合考虑三边比例和角度差异。
    """
    ratio1 = np.sort(btc1.triangle_sides)
    ratio2 = np.sort(btc2.triangle_sides)

    # 归一化边长
    ratio1 = ratio1 / (ratio1.sum() + 1e-12)
    ratio2 = ratio2 / (ratio2.sum() + 1e-12)
    side_sim = 1.0 - np.sum(np.abs(ratio1 - ratio2)) / 3.0

    # 角度相似度
    ang1 = np.sort(btc1.angles)
    ang2 = np.sort(btc2.angles)
    ang_diff = np.abs(ang1 - ang2) / np.pi
    angle_sim = 1.0 - np.sum(ang_diff) / 3.0

    return 0.5 * side_sim + 0.5 * angle_sim


def estimate_relative_pose(btc_query, btc_candidate):
    """
    从两个匹配的 BTC 估计相对位姿。
    使用三角形的三个顶点对应关系计算 3D-3D 变换。
    返回 (R 3x3, t 3x1) 或 None。
    """
    src_pts = np.array([
        btc_query.desc_a.pose_3d,
        btc_query.desc_b.pose_3d,
        btc_query.desc_c.pose_3d
    ])
    dst_pts = np.array([
        btc_candidate.desc_a.pose_3d,
        btc_candidate.desc_b.pose_3d,
        btc_candidate.desc_c.pose_3d
    ])

    # Umeyama 算法估计相似变换
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = LA.svd(H)

    R = Vt.T @ U.T
    if LA.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    return R, t


def search_loop_btc(query_btcs, database, query_frame_id, num_exclude_recent=NUM_EXCLUDE_RECENT):
    """
    在 BTC 数据库中搜索回环。
    返回 (best_cand_frame, best_score, relative_pose, yaw_deg) 或 None。
    """
    best_cand_frame = -1
    best_score = 0.0
    best_pose = None

    # 收集候选帧
    candidate_scores = defaultdict(float)
    candidate_pairs = defaultdict(list)

    for q_btc in query_btcs:
        key = btc_hash_key(q_btc)
        if key not in database:
            # 也尝试近似的 key
            found = False
            vx, vy, vz, side_code, angle_code = key
            for dvx in [-1, 0, 1]:
                for dvy in [-1, 0, 1]:
                    for dvz in [-1, 0, 1]:
                        alt_key = (vx + dvx, vy + dvy, vz + dvz, side_code, angle_code)
                        if alt_key in database:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                key = alt_key
            else:
                continue

        for db_btc in database.get(key, []):
            cand_frame = db_btc.frame_id
            if query_frame_id - cand_frame < num_exclude_recent:
                continue

            sim = compute_btc_similarity(q_btc, db_btc)
            if sim > 0.3:  # 基本相似度要求
                candidate_scores[cand_frame] = max(candidate_scores[cand_frame], sim)
                candidate_pairs[cand_frame].append((q_btc, db_btc, sim))

    if not candidate_scores:
        return None

    # 找到最佳候选帧
    sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])
    best_cand_frame, best_score = sorted_cands[0]

    if best_score < BTC_MATCH_SCORE_THRES:
        return None

    # 用最佳匹配对估计相对位姿
    best_pair = max(candidate_pairs[best_cand_frame], key=lambda x: x[2])
    q_btc, db_btc, _ = best_pair

    pose_result = estimate_relative_pose(q_btc, db_btc)
    if pose_result is None:
        return None

    R, t = pose_result
    # 计算偏航角
    yaw = np.arctan2(R[1, 0], R[0, 0])

    relative_pose = np.eye(4)
    relative_pose[:3, :3] = R
    relative_pose[:3, 3] = t

    return (best_cand_frame, best_score, relative_pose, np.degrees(yaw))


# ======================== GICP 验证 ========================

def verify_loop_gicp(source_pts, target_pts, init_pose):
    """
    使用 Open3D 的 GICP 进行回环验证。
    返回 (refined_pose_4x4, fitness_score) 或 None。
    """
    if not HAS_O3D:
        return None

    if len(source_pts) < 100 or len(target_pts) < 100:
        return None

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)
    target.points = o3d.utility.Vector3dVector(target_pts)

    # 使用 GICP 进行精化对齐
    gicp_estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    gicp_estimation.epsilon = GICP_EPSILON

    try:
        result = o3d.pipelines.registration.registration_generalized_icp(
            source, target,
            GICP_MAX_DIST,
            init_pose,
            gicp_estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=GICP_ITERATIONS,
                relative_fitness=1e-4,
                relative_rmse=1e-4
            )
        )

        if result.fitness > 0 and result.inlier_rmse < GICP_FITNESS_THRES:
            return result.transformation, result.inlier_rmse
        else:
            return None

    except Exception as e:
        print(f"         GICP 异常: {e}")
        return None


# ======================== 位姿图优化 ========================

def optimize_pose_graph(poses, loop_pairs):
    """
    使用 GTSAM 进行位姿图优化。
    loop_pairs: [(frame_a, frame_b, rel_pose_4x4, score), ...]
    """
    if not HAS_GTSAM:
        print("[SKIP] GTSAM 不可用，跳过位姿图优化。")
        return None

    print("\n[ 使用 GTSAM 进行位姿图优化 ]")

    def to_gtsam_pose3(T):
        R = T[:3, :3]
        t = T[:3, 3]
        q = gtsam.Rot3.RzRyRx(
            np.arctan2(R[2, 1], R[2, 2]),
            np.arcsin(-R[2, 0]),
            np.arctan2(R[1, 0], R[0, 0])
        )
        return gtsam.Pose3(q, gtsam.Point3(t[0], t[1], t[2]))

    def from_gtsam_pose3(p):
        t = p.translation()
        R = p.rotation().matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t[0], t[1], t[2]]
        return T

    prior_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
    )
    odom_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
    )
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Variances(
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        )
    )

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pose0 = to_gtsam_pose3(poses[0])
    graph.add(gtsam.PriorFactorPose3(0, pose0, prior_noise))
    initial.insert(0, pose0)

    for i in range(1, len(poses)):
        pose_i = to_gtsam_pose3(poses[i])
        initial.insert(i, pose_i)

        T_delta = np.linalg.inv(poses[i - 1]) @ poses[i]
        delta = to_gtsam_pose3(T_delta)
        graph.add(gtsam.BetweenFactorPose3(i - 1, i, delta, odom_noise))

    loop_added = 0
    for a, b, rel_pose, score in loop_pairs:
        rel_gtsam = to_gtsam_pose3(rel_pose)
        graph.add(gtsam.BetweenFactorPose3(int(a), int(b), rel_gtsam, loop_noise))
        loop_added += 1

    print(f"      图: {len(poses)} 个节点, "
          f"{len(poses) - 1 + loop_added} 个因子 "
          f"(里程计={len(poses) - 1}, 回环={loop_added})")

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    print(f"      优化完成。最终误差: {optimizer.error():.6f}")

    opt_poses = []
    for i in range(len(poses)):
        p = result.atPose3(i)
        opt_poses.append(from_gtsam_pose3(p))

    return opt_poses


# ======================== 主程序 ========================

def main():
    parser = argparse.ArgumentParser(
        description="Super-LIO 离线回环检测工具 (BTC + GICP)"
    )
    parser.add_argument(
        "data_dir", type=str, nargs='?', default=DEFAULT_DATA_DIR,
        help=f"数据目录，包含 odom_poses.txt 和 Scans/ (默认: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--btc-match-thres", type=float, default=BTC_MATCH_SCORE_THRES,
        help=f"BTC 匹配分数阈值 (默认: {BTC_MATCH_SCORE_THRES})"
    )
    parser.add_argument(
        "--no-gicp", action="store_true",
        help="禁用 GICP 回环验证"
    )
    parser.add_argument(
        "--skip-optimization", action="store_true",
        help="跳过位姿图优化（仅检测回环）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认与 data_dir 相同）"
    )
    parser.add_argument(
        "--save-map", action="store_true",
        help="保存合并后的点云地图（需要 open3d）"
    )

    args = parser.parse_args()

    data_dir = args.data_dir.rstrip('/')
    output_dir = args.output_dir.rstrip('/') if args.output_dir else data_dir
    os.makedirs(output_dir, exist_ok=True)

    btc_match_thres = args.btc_match_thres
    enable_gicp = not args.no_gicp and HAS_O3D
    skip_optimization = args.skip_optimization
    save_map = args.save_map and HAS_O3D

    print("=" * 70)
    print("  Super-LIO 离线回环检测 (BTC + GICP)")
    print("=" * 70)
    print(f"  数据目录      : {data_dir}")
    print(f"  输出目录      : {output_dir}")
    print(f"  BTC 匹配阈值  : {btc_match_thres}")
    print(f"  GICP 验证     : {'启用' if enable_gicp else '禁用'}")
    print(f"  图优化        : {'启用' if HAS_GTSAM and not skip_optimization else '跳过'}")
    print(f"  保存地图      : {'是' if save_map else '否'}")
    print("=" * 70)

    t_start = time.time()

    # ==============================================================
    # 步骤 1：加载里程计位姿
    # ==============================================================
    odom_file = os.path.join(data_dir, "odom_poses.txt")
    if not os.path.exists(odom_file):
        print(f"[ERROR] 未找到 odom_poses.txt: {odom_file}")
        sys.exit(1)

    print(f"\n[1/5] 加载里程计位姿 ... ", end="", flush=True)
    poses = load_poses(odom_file)
    print(f"已加载 {len(poses)} 个位姿")

    # ==============================================================
    # 步骤 2：加载关键帧点云
    # ==============================================================
    scan_dir = os.path.join(data_dir, "Scans")
    if not os.path.isdir(scan_dir):
        print(f"[ERROR] 未找到 Scans 目录: {scan_dir}")
        sys.exit(1)

    print(f"[2/5] 加载关键帧点云 ... ", end="", flush=True)
    if HAS_O3D:
        scans = load_scans_o3d(scan_dir, len(poses))
    else:
        print("\n[ERROR] 需要 open3d 来读取 PCD 文件。")
        sys.exit(1)

    n_valid = sum(1 for s, _ in scans if len(s) > 0)
    print(f"已加载 {n_valid}/{len(poses)} 个含有点云的关键帧")

    # ==============================================================
    # 步骤 3：生成 BTC 描述子并构建数据库
    # ==============================================================
    print(f"\n[3/5] 生成 BTC 描述子 ...")
    btc_database = defaultdict(list)  # hash_key -> [BTC, ...]
    all_frame_btcs = []  # [(frame_id, [BTC, ...]), ...]

    for i in range(len(poses)):
        pts, _ = scans[i]
        if len(pts) < BTC_PLANE_MIN_POINTS:
            all_frame_btcs.append((i, []))
            continue

        btcs = generate_btcs_for_frame(pts, i)
        all_frame_btcs.append((i, btcs))

        # 添加到数据库
        for btc in btcs:
            key = btc_hash_key(btc)
            btc_database[key].append(btc)

        if (i + 1) % 200 == 0:
            total_btcs = sum(len(btcs) for _, btcs in all_frame_btcs)
            print(f"      已处理 {i + 1}/{len(poses)} 帧, "
                  f"描述子总数: {total_btcs}")

    total_btcs = sum(len(btcs) for _, btcs in all_frame_btcs)
    print(f"      完成: {len(poses)} 帧, {total_btcs} 个 BTC, "
          f"数据库条目: {len(btc_database)}")

    # ==============================================================
    # 步骤 4：检测回环
    # ==============================================================
    print(f"\n[4/5] 检测回环 (BTC 搜索) ...")

    loop_candidates = []  # [(cand_idx, curr_idx, btc_score, relative_pose, yaw_deg), ...]

    for frame_id, btcs in all_frame_btcs:
        if frame_id < NUM_EXCLUDE_RECENT + 1:
            continue
        if len(btcs) == 0:
            continue

        result = search_loop_btc(btcs, btc_database, frame_id)
        if result is not None:
            cand_frame, score, rel_pose, yaw_deg = result
            if score >= btc_match_thres:
                loop_candidates.append((cand_frame, frame_id, score, rel_pose, yaw_deg))
                print(f"      [候选] 帧 {cand_frame:6d} <-> {frame_id:6d}, "
                      f"BTC 分数={score:.4f}, 偏航={yaw_deg:+.1f}deg")

    print(f"      发现 {len(loop_candidates)} 个回环候选")
    if len(loop_candidates) == 0:
        print("\n[结果] 未检测到回环。尝试降低 --btc-match-thres 阈值。")

    # ==============================================================
    # 步骤 4b：GICP 验证
    # ==============================================================
    verified_loops = []  # [(frame_a, frame_b, btc_score, rel_pose, fitness_score), ...]

    if enable_gicp and loop_candidates:
        print(f"\n      --- GICP 验证 ---")
        for a, b, btc_score, rel_pose, yaw_deg in loop_candidates:
            source_pts = scans[b][0]  # 当前帧（被变换）
            target_pts = scans[a][0]  # 历史帧（目标）

            if len(source_pts) < 100 or len(target_pts) < 100:
                print(f"         跳过 {a}<->{b}: 点数过少")
                continue

            print(f"      验证中: 帧 {a} <-> 帧 {b} ... ", end="", flush=True)
            gicp_result = verify_loop_gicp(source_pts, target_pts, rel_pose)

            if gicp_result is not None:
                refined_pose, fitness = gicp_result
                verified_loops.append((a, b, btc_score, refined_pose, fitness))
                print(f"通过 (score={fitness:.4f})")
            else:
                print(f"拒绝 (GICP 不收敛)")

        print(f"\n      已验证回环: {len(verified_loops)}/{len(loop_candidates)}")
    else:
        # 没有 GICP 时，直接用 BTC 给出的相对位姿
        for a, b, btc_score, rel_pose, yaw_deg in loop_candidates:
            verified_loops.append((a, b, btc_score, rel_pose, btc_score))

    # ==============================================================
    # 步骤 5：位姿图优化与输出
    # ==============================================================
    print(f"\n[5/5] 保存结果 ...")

    # 保存回环对列表
    loop_file = os.path.join(output_dir, "loop_pairs.txt")
    save_loop_pairs(loop_file, [
        (a, b, btc_score, fit_score)
        for a, b, btc_score, _, fit_score in verified_loops
    ])
    print(f"      已保存回环对: {loop_file}")

    # 保存回环详情
    detail_file = os.path.join(output_dir, "loop_details.txt")
    with open(detail_file, 'w') as f:
        f.write(f"# BTC + GICP 回环检测结果\n")
        f.write(f"# GICP 验证: {'启用' if enable_gicp else '禁用'}\n")
        f.write(f"# 总帧数: {len(poses)}, 回环候选: {len(loop_candidates)}, "
                f"已验证: {len(verified_loops)}\n")
        f.write(f"# 生成时间: {time.ctime()}\n")
        f.write(f"# 格式: frame_a frame_b btc_score fitness_score\n")
        for a, b, btc_score, _, fit_score in verified_loops:
            f.write(f"{a} {b} {btc_score:.6f} {fit_score:.6f}\n")
    print(f"      已保存回环详情: {detail_file}")

    # 位姿图优化
    opt_poses = None
    if HAS_GTSAM and not skip_optimization:
        if len(verified_loops) > 0:
            # 转换为优化需要的格式
            loop_data = [(a, b, rel_pose, fit_score)
                        for a, b, _, rel_pose, fit_score in verified_loops]
            opt_poses = optimize_pose_graph(poses, loop_data)
        else:
            print("      未检测到回环，将里程计位姿复制为优化后位姿。")
            opt_poses = poses
    elif skip_optimization:
        print("      位姿图优化已跳过（--skip-optimization）。")
    else:
        print("      GTSAM 不可用，将里程计位姿复制为优化后位姿。")
        opt_poses = poses

    if opt_poses is not None:
        opt_file = os.path.join(output_dir, "optimized_poses.txt")
        save_poses_kitti(opt_file, opt_poses)
        print(f"      已保存优化后位姿: {opt_file}")

    # 保存合并地图
    if save_map:
        print(f"      正在生成合并地图 ...")
        map_poses = opt_poses if opt_poses is not None else poses

        merged_pcd = o3d.geometry.PointCloud()
        for i in range(len(scans)):
            pts_i, _ = scans[i]
            if len(pts_i) == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_i)
            pcd.transform(map_poses[i])
            merged_pcd += pcd

        merged_pcd = merged_pcd.voxel_down_sample(0.1)

        map_file = os.path.join(output_dir, "merged_map.pcd")
        o3d.io.write_point_cloud(map_file, merged_pcd)
        print(f"      已保存合并地图: {map_file} ({len(merged_pcd.points)} 个点)")

    # ==============================================================
    # 汇总
    # ==============================================================
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  总帧数        : {len(poses)}")
    print(f"  回环候选数    : {len(loop_candidates)}")
    print(f"  已验证回环数  : {len(verified_loops)}")
    print(f"  运行时间      : {t_elapsed:.1f}s")
    print(f"  输出目录      : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
