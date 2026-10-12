#!/usr/bin/env python3
"""
回环检测公共模块 — 与在线 C++ 版本流程完全一致

提供:
  - GICP 精化 (对应 C++ GICPRegistration)
  - 回环验证 (对应 C++ validateLoopClosure)
  - 位姿图优化 (对应 C++ GTSAM ISAM2 增量优化)
  - 关键帧选择 (对应 C++ keyframeMeterGap / keyframeRadGap)
  - 完整离线回环流程 (OfflineLoopCloser)
"""

import os
import re
import time
import concurrent.futures
import threading
import numpy as np
from numpy import linalg as LA
import yaml

# ROS2 可选导入（用于离线脚本输出话题，供 RViz 比较 PGO 效果）
HAS_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Path as RosPath, Odometry as RosOdometry
    from geometry_msgs.msg import PoseStamped as RosPoseStamped
    from visualization_msgs.msg import Marker as RosMarker
    from visualization_msgs.msg import MarkerArray as RosMarkerArray
    from builtin_interfaces.msg import Duration as RosDuration
    from builtin_interfaces.msg import Time as RosTime
    HAS_ROS2 = True
except ImportError:
    pass

# pclomp GICP_OMP (与在线定位同款实现)
try:
    import gicp_omp_cpp
    HAS_GICP_OMP = True
except ImportError:
    HAS_GICP_OMP = False
    print("[WARN] gicp_omp_cpp 未安装，GICP配准功能已禁用。")

# BTC C++ 模块
try:
    import btc_cpp
    HAS_BTC_CPP = True
except ImportError:
    HAS_BTC_CPP = False
    print("[WARN] btc_cpp 未安装，BTC回环检测功能已禁用。")

# ScanContext C++ 模块
try:
    import sc_cpp
    HAS_SC_CPP = True
except ImportError:
    HAS_SC_CPP = False
    print("[WARN] sc_cpp 未安装，ScanContext回环检测功能已禁用。")

# GTSAM位姿图优化
try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    print("[WARN] gtsam 未安装，位姿图优化功能已禁用。安装: pip install gtsam")
    print("[ERROR] gicp_omp_cpp 未安装，GICP 功能不可用。请重新编译: colcon build --packages-select sc_pgo_ros2")

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    print("[WARN] gtsam 未安装，位姿图优化功能已禁用。安装: pip install gtsam")

from btc_common import (
    down_sampling_voxel,  # 点云下采样函数（仍然需要）
)


# ======================== GICP 配置 (对应 C++ GICPConfig) ========================

class GICPConfig:
    """对应 C++ GICPConfig，默认值适配 Mid360 回环场景"""

    def __init__(self):
        # 所有默认值与 btc_config.yaml 中 gicp 部分一致
        self.transformation_epsilon = 0.001        # 收敛精度（变换增量阈值）
        self.max_correspondence_distance = 5.0     # 最大对应点距离 (m)
        self.rotation_epsilon = 0.002
        self.k_correspondences = 20
        self.max_optimizer_iterations = 20
        self.gicp_epsilon = 0.00001                # GICP 协方差正则化（防奇异）
        self.max_iterations = 64                   # 每级最大迭代次数
        self.fitness_score_threshold = 0.25        # GICP fitness 阈值
        self.num_threads = 4
        self.scan_ds_size = 0.1                    # GICP配准前点云下采样体素 (m)
        # 多级配准参数
        self.coarse_ds_size = 0.25                 # 粗配准下采样体素
        self.coarse_max_iter = 50                  # 粗配准最大迭代
        self.coarse_max_dist = 10.0                # 粗配准最大对应距离
        self.max_init_translation = 15.0           # 初始平移超过此值时跳过验证


class GICPResult:
    """对应 C++ GICPResult"""
    def __init__(self):
        self.transformation = np.eye(4)
        self.has_converged = False
        self.fitness_score = float('inf')      # mean_inlier_dist (m)，越小越好
        self.overlap_ratio = 0.0               # 0~1，越大越好
        self.num_iterations = 0
        self.coarse_iterations = 0


class _GicpStats:
    """GICP 调用次数/耗时统计（线程安全），用于离线性能诊断"""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.calls = 0
            self.total_s = 0.0
            self.max_s = 0.0
            self.src_pts_total = 0
            self.tgt_pts_total = 0
            self.by_tag = {}          # tag -> {'calls':int, 'total_s':float}
            self.hist = []            # 明细: (耗时s, tag, src_pts, tgt_pts)
            self.record_detail = True

    def add(self, dt, tag, n_src, n_tgt, iters=0, success=False):
        with self._lock:
            self.calls += 1
            self.total_s += dt
            if dt > self.max_s:
                self.max_s = dt
            self.src_pts_total += n_src
            self.tgt_pts_total += n_tgt
            e = self.by_tag.setdefault(tag, {'calls': 0, 'total_s': 0.0})
            e['calls'] += 1
            e['total_s'] += dt
            if self.record_detail:
                self.hist.append((dt, tag, n_src, n_tgt, iters, success))

    def snapshot(self):
        """返回 (调用次数, 累计耗时s)，供运行中实时打印"""
        with self._lock:
            return self.calls, self.total_s

    def report(self, title="GICP 耗时统计"):
        with self._lock:
            print(f"\n  ----- {title} -----")
            if self.calls == 0:
                print("  无 GICP 调用")
                return
            print(f"  调用总次数   : {self.calls}")
            print(f"  总计耗时     : {self.total_s:.2f}s")
            print(f"  平均单次     : {self.total_s / self.calls:.3f}s")
            print(f"  最慢单次     : {self.max_s:.3f}s")
            print(f"  平均点云规模 : src={self.src_pts_total / self.calls:.0f}, "
                  f"tgt={self.tgt_pts_total / self.calls:.0f}")
            for tag, e in sorted(self.by_tag.items(), key=lambda kv: -kv[1]['total_s']):
                print(f"    - {tag:<12}: {e['calls']:>5} 次, {e['total_s']:>9.2f}s "
                      f"(均 {e['total_s'] / e['calls']:.3f}s)")
            if self.hist:
                print("  最慢的 5 次:")
                for dt, tag, n_src, n_tgt, iters, succ in sorted(self.hist, reverse=True)[:5]:
                    print(f"    {dt:>7.3f}s  {tag:<12} src={n_src:>7} tgt={n_tgt:>7} "
                          f"iter={iters:>4} {'OK' if succ else 'FAIL'}")
                self._report_iterations()

    def _report_iterations(self):
        """按 成功/失配 分组的精配准迭代次数分布，用于确定 fine_max_iter 上限"""
        ok = [h[4] for h in self.hist if h[5]]
        bad = [h[4] for h in self.hist if not h[5]]
        print("  精配准迭代次数分布 (成功匹配 vs 失配):")
        for name, arr in (("成功匹配", ok), ("失配", bad)):
            if not arr:
                print(f"    - {name}: 无")
                continue
            a = np.array(arr, dtype=np.int32)
            print(f"    - {name}: n={len(arr):<4} min={a.min():<4} "
                  f"中位={int(np.median(a)):<4} p90={int(np.percentile(a, 90)):<4} max={a.max()}")
        if ok:
            print(f"    => 建议 gicp_max_iterations ≈ {max(ok) + 2} "
                  f"(成功匹配最大 {max(ok)} + 余量；超过的失配迭代属于过度计算)")


GICP_STATS = _GicpStats()


# ---- GICP_OMP Manager 单例（避免每次配准都重新构造 Manager 并打印"初始化完成"）----
# alignTwoStage 内部为每次调用单独创建 coarse/fine GICP 对象（线程安全），
# Manager 自身仅持有只读的配置标量，可安全跨线程共享，因此全局复用一个实例。
_GICP_MGR = None
_GICP_MGR_SIG = None
_GICP_MGR_LOCK = threading.Lock()


def _gicp_config_sig(c):
    """GICPConfig 中与 setConfig 相关的字段签名，用于判断是否需要重新配置"""
    return (
        c.max_correspondence_distance, c.transformation_epsilon, c.rotation_epsilon,
        c.k_correspondences, c.max_optimizer_iterations, c.gicp_epsilon,
        c.max_iterations, c.num_threads,
        getattr(c, 'coarse_ds_size', None), getattr(c, 'coarse_max_iter', None),
        getattr(c, 'coarse_max_dist', None),
    )


def _get_gicp_manager(config):
    """返回复用的 GicpOmpManager 单例；仅首次或配置变化时调用 setConfig。"""
    global _GICP_MGR, _GICP_MGR_SIG
    cfg_dict = {
        'max_correspondence_distance': config.max_correspondence_distance,
        'transformation_epsilon': config.transformation_epsilon,
        'rotation_epsilon': config.rotation_epsilon,
        'k_correspondences': config.k_correspondences,
        'max_optimizer_iterations': config.max_optimizer_iterations,
        'gicp_epsilon': config.gicp_epsilon,
        'max_iterations': config.max_iterations,
        'num_threads': config.num_threads,
    }
    with _GICP_MGR_LOCK:
        if _GICP_MGR is None:
            _GICP_MGR = gicp_omp_cpp.GicpOmpManager()
            _GICP_MGR.setConfig(cfg_dict)
            _GICP_MGR_SIG = _gicp_config_sig(config)
        elif _gicp_config_sig(config) != _GICP_MGR_SIG:
            _GICP_MGR.setConfig(cfg_dict)
            _GICP_MGR_SIG = _gicp_config_sig(config)
        return _GICP_MGR


def gicp_align(source_pts, target_pts, initial_guess=None, config=None,
               compute_overlap=False):
    """
    pclomp GICP_OMP 配准 — 与在线 C++ 定位使用完全相同的 pclomp::GeneralizedIterativeClosestPoint。

    - fitness_score 量级与在线一致（0.01~0.05 级别）
    - 协方差计算 + KDTree+BFGS 均 OpenMP 并行
    - 内部自动从 KNN 邻居计算协方差矩阵，无需手动估算法向量

    source_pts: Nx3 numpy array (或 Nx4 with intensity)
    target_pts: Nx3 numpy array (或 Nx4 with intensity)
    initial_guess: 4x4 numpy array
    config: GICPConfig
    compute_overlap: bool, 是否额外计算 overlap_ratio（需再建一棵KDTree并全量近邻搜索，
                     仅 save_fused_scans 时才需要，默认关闭以加速）
    返回: GICPResult
    """
    if not HAS_GICP_OMP:
        print("[GICP] gicp_omp_cpp 不可用，跳过")
        return None

    if config is None:
        config = GICPConfig()

    if initial_guess is None:
        initial_guess = np.eye(4)

    if len(source_pts) < 100 or len(target_pts) < 100:
        print("[GICP] 点云点数不足，跳过")
        return None

    # 检查初始猜测矩阵是否有效（避免NaN/Inf导致C++崩溃）
    if not np.all(np.isfinite(initial_guess)):
        print("[GICP] 初始猜测矩阵包含NaN/Inf，跳过")
        return None

    # Nx3 -> Nx4 (添加 intensity=0)
    if source_pts.shape[1] == 3:
        src_nx4 = np.hstack([source_pts.astype(np.float32), np.zeros((len(source_pts), 1), dtype=np.float32)])
    else:
        src_nx4 = source_pts.astype(np.float32)
    if target_pts.shape[1] == 3:
        tgt_nx4 = np.hstack([target_pts.astype(np.float32), np.zeros((len(target_pts), 1), dtype=np.float32)])
    else:
        tgt_nx4 = target_pts.astype(np.float32)

    result = GICPResult()

    try:
        gicp = _get_gicp_manager(config)

        res = gicp.alignTwoStage(
            src_nx4, tgt_nx4, initial_guess.astype(np.float64),
            coarse_ds_size=config.coarse_ds_size,
            coarse_max_iter=config.coarse_max_iter,
            coarse_max_dist=config.coarse_max_dist,
            fine_max_iter=config.max_iterations,
            fine_max_dist=config.max_correspondence_distance,
            compute_overlap=compute_overlap,
        )

        result.transformation = np.asarray(res['transformation'])
        result.has_converged = bool(res['has_converged'])
        result.fitness_score = float(res['fitness_score'])          # mean_inlier_dist (m)
        result.overlap_ratio = float(res.get('overlap_ratio', 0.0))
        result.num_iterations = int(res.get('iterations', 0))
        result.coarse_iterations = int(res.get('coarse_iterations', 0))

    except Exception as e:
        print(f"[GICP] 异常: {e}")
        import traceback
        traceback.print_exc()
        return None

    return result


def check_degeneracy(source_pts, target_pts, transformation, max_correspondence_distance=2.0):
    """
    检查 GICP 配准是否存在退化方向（对应问题3：退化方向校验）

    通过分析配准后的残差分布来估计退化方向：
    - 在长廊等结构单调环境中，GICP 可能沿某个方向产生错误漂移
    - 通过 PCA 分析对应点残差，检测是否某个方向退化

    source_pts: Nx3 numpy array (源点云)
    target_pts: Nx3 numpy array (目标点云)
    transformation: 4x4 numpy array (配准变换矩阵)
    max_correspondence_distance: float (对应点搜索距离)

    返回: (is_degenerate, degeneracy_direction, eigenvalues)
        - is_degenerate: bool (是否退化)
        - degeneracy_direction: np.array (退化方向，3维向量)
        - eigenvalues: np.array (特征值，用于调试)
    """
    from scipy.spatial import cKDTree

    if len(source_pts) < 100 or len(target_pts) < 100:
        return False, np.zeros(3), np.zeros(3)

    # 变换源点云
    ones = np.ones((len(source_pts), 1))
    source_homo = np.hstack([source_pts, ones])
    transformed = (transformation @ source_homo.T).T[:, :3]

    # 建立KD树查找对应点
    kdtree = cKDTree(target_pts)

    # 收集残差向量
    residuals = []
    for pt in transformed:
        # 找最近点
        dist, idx = kdtree.query(pt, k=1)
        if dist < max_correspondence_distance:
            target_pt = target_pts[idx]
            residual = pt - target_pt
            residuals.append(residual)

    if len(residuals) < 100:
        return False, np.zeros(3), np.zeros(3)

    residuals = np.array(residuals)  # Nx3

    # PCA 分析残差分布
    mean = residuals.mean(axis=0)
    residuals_centered = residuals - mean
    cov = residuals_centered.T @ residuals_centered / len(residuals)

    # 特征值分解
    eigenvalues, eigenvectors = LA.eigh(cov)
    eigenvalues = eigenvalues[::-1]  # 降序
    eigenvectors = eigenvectors[:, ::-1]

    # 检测退化：如果最小特征值远小于其他特征值（数量级差异）
    # 说明某个方向的残差约束很弱，可能退化
    min_eigenvalue = eigenvalues[2]
    max_eigenvalue = eigenvalues[0]

    # 退化判定：最小特征值 < 最大特征值的 1/10
    is_degenerate = min_eigenvalue < max_eigenvalue * 0.1

    degeneracy_direction = eigenvectors[:, 2] if is_degenerate else np.zeros(3)

    if is_degenerate:
        print(f"[Degeneracy] 检测到退化! 特征值: [{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}]")
        print(f"[Degeneracy] 退化方向: [{degeneracy_direction[0]:.3f}, {degeneracy_direction[1]:.3f}, {degeneracy_direction[2]:.3f}]")

    return is_degenerate, degeneracy_direction, eigenvalues


# ======================== 回环验证 (对应 C++ validateLoopClosure) ========================

def validate_loop_closure(pose_prev, pose_curr, relative_pose,
                          max_loop_distance=100.0, max_yaw_diff=None):
    """
    对应 C++ validateLoopClosure

    pose_prev, pose_curr: 4x4 numpy array (关键帧位姿)
    relative_pose: 4x4 numpy array (BTC/GICP 估计的相对位姿)
    max_loop_distance: 最大回环距离 (C++ 中为 100m)
    max_yaw_diff: 最大偏航角差 (C++ 中为 0.75π)

    返回: bool
    """
    if max_yaw_diff is None:
        max_yaw_diff = np.pi * 0.75

    # 计算两个关键帧之间的位姿差
    pose_diff = np.linalg.inv(pose_prev) @ pose_curr

    # 提取平移距离
    distance = LA.norm(pose_diff[:3, 3])
    if distance > max_loop_distance:
        print(f"[Loop validation] 距离过大: {distance:.2f} > {max_loop_distance}, 拒绝回环")
        return False

    # 提取偏航角差
    yaw_diff = abs(_extract_yaw(pose_diff))
    if yaw_diff > max_yaw_diff:
        print(f"[Loop validation] 偏航角差过大: {yaw_diff:.2f} > {max_yaw_diff:.2f}, 拒绝回环")
        return False

    print(f"[Loop validation] 通过. 距离: {distance:.2f}, 偏航角差: {yaw_diff:.2f}")
    return True


def _extract_yaw(T):
    """从 4x4 变换矩阵提取偏航角"""
    R = T[:3, :3]
    return np.arctan2(R[1, 0], R[0, 0])


def _interp_rotation_matrix(R1, R2, t):
    """球面线性插值（Slerp）两个旋转矩阵"""
    # R1 到 R2 的相对旋转
    R_rel = R1.T @ R2
    # 提取旋转轴和角度
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    if abs(angle) < 1e-6:
        return R1.copy()
    axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                     R_rel[0, 2] - R_rel[2, 0],
                     R_rel[1, 0] - R_rel[0, 1]])
    axis = axis / np.linalg.norm(axis)
    # Rodrigues 公式插值
    theta = angle * t
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    R_interp = np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R1 @ R_interp


# ======================== 位姿图优化 (对应 C++ GTSAM ISAM2) ========================

def init_noises(loop_kernel='huber', loop_k=0.0):
    """
    对应 C++ initNoises
    返回 (prior_noise, odom_noise, robust_loop_noise, robust_gps_noise)

    噪声模型说明（odom 可信场景）:
      - prior_noise: 先验噪声极小，固定首个位姿
      - odom_noise: 里程计噪声极小（sigma平移1cm, 旋转0.5°），坚信odom
      - loop_noise: 回环噪声较大（sigma平移1m, 旋转0.5rad），不信任回环
        配合鲁棒核，仅当回环与 odom 高度一致时才起作用

    loop_kernel: 'huber' | 'cauchy' | 'none'
      - Huber: 对中等离群稳健，抑制力温和
      - Cauchy: 对大幅度离群衰减更快，误回环(如多楼层楼梯)抑制更强
    loop_k: 核函数尺度参数；<=0 时按核类型取默认 (huber=1.345, cauchy=1.0)
    """
    prior_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
    )
    # 坚信 odom: sigma平移=0.01m, sigma旋转=0.008rad (≈0.5°)
    odom_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-4, 1e-4, 1e-4, 6.4e-5, 6.4e-5, 6.4e-5])
    )
    # 不信任回环: sigma平移=1m, sigma旋转=0.5rad
    loop_noise_vector = np.array([1.0, 1.0, 1.0, 0.25, 0.25, 0.25])
    loop_base = gtsam.noiseModel.Diagonal.Variances(loop_noise_vector)
    _kernel = (loop_kernel or 'none').lower()
    if _kernel == 'cauchy':
        _k = loop_k if loop_k > 0 else 1.0
        robust_loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(_k), loop_base)
    elif _kernel == 'huber':
        _k = loop_k if loop_k > 0 else 1.345
        robust_loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(_k), loop_base)
    else:
        robust_loop_noise = loop_base
    # GPS noise
    big_noise_tolerant_to_xy = 1e9
    gps_altitude_noise = 250.0
    robust_gps_vector = np.array([big_noise_tolerant_to_xy, big_noise_tolerant_to_xy, gps_altitude_noise])
    robust_gps_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1),
        gtsam.noiseModel.Diagonal.Variances(robust_gps_vector)
    )
    return prior_noise, odom_noise, robust_loop_noise, robust_gps_noise


def pose6d_to_gtsam_pose3(pose6d):
    """对应 C++ Pose6DtoGTSAMPose3"""
    return gtsam.Pose3(
        gtsam.Rot3.RzRyRx(pose6d[3], pose6d[4], pose6d[5]),
        gtsam.Point3(pose6d[0], pose6d[1], pose6d[2])
    )


def matrix_to_gtsam_pose3(T):
    """4x4 变换矩阵 -> gtsam.Pose3"""
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t[0], t[1], t[2]))


def gtsam_pose3_to_matrix(pose3):
    """gtsam.Pose3 -> 4x4 变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = pose3.rotation().matrix()
    T[:3, 3] = pose3.translation()
    return T


# ======================== 关键帧选择 (对应 C++ process_pg 中的关键帧判断) ========================

def is_keyframe(pose_prev, pose_curr, keyframe_meter_gap=5.0, keyframe_deg_gap=10.0):
    """
    对应 C++ process_pg 中的关键帧判断逻辑:
      translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap

    pose_prev, pose_curr: (x, y, z, roll, pitch, yaw) 元组
    返回: bool
    """
    T_prev = _pose6d_to_matrix(pose_prev)
    T_curr = _pose6d_to_matrix(pose_curr)
    T_delta = np.linalg.inv(T_prev) @ T_curr

    dx = abs(T_delta[0, 3])
    dy = abs(T_delta[1, 3])
    dz = abs(T_delta[2, 3])
    delta_translation = np.sqrt(dx**2 + dy**2 + dz**2)

    roll, pitch, yaw = _extract_rpy(T_delta)
    delta_rotation = abs(roll) + abs(pitch) + abs(yaw)

    keyframe_rad_gap = np.deg2rad(keyframe_deg_gap)
    return delta_translation > keyframe_meter_gap or delta_rotation > keyframe_rad_gap


def _pose6d_to_matrix(pose6d):
    """(x,y,z,roll,pitch,yaw) -> 4x4"""
    T = np.eye(4)
    R = _rpy_to_matrix(pose6d[3], pose6d[4], pose6d[5])
    T[:3, :3] = R
    T[:3, 3] = [pose6d[0], pose6d[1], pose6d[2]]
    return T


def _rpy_to_matrix(roll, pitch, yaw):
    """RPY -> 3x3 旋转矩阵"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return R


def _quat_xyzw_to_matrix(qx, qy, qz, qw):
    """四元数 (x,y,z,w) -> 3x3 旋转矩阵（自动归一化）"""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ])


def _extract_rpy(T):
    """4x4 -> (roll, pitch, yaw)"""
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _rot_angle_rad(R):
    """3x3 旋转矩阵 -> 旋转角 (rad)"""
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _far_range_points(cloud, far_dist):
    """提取距传感器原点 > far_dist 的点(窗外/远景)，用于跨楼层共同景观对齐。

    关键帧点云为 body 系，故点范数即为量程。
    """
    if cloud is None or len(cloud) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.linalg.norm(cloud[:, :3], axis=1)
    return cloud[rng > far_dist]


def _is_ros2_param_format(data):
    """判断 yaml 内容是否为 ROS2 参数格式（/**: ros__parameters: 嵌套）"""
    return isinstance(data, dict) and '/**' in data


# C++ load_config_setting (cv::FileStorage) 需要的 BTC 顶层键
_BTC_CPP_KEYS = [
    'useful_corner_num', 'plane_merge_normal_thre', 'plane_merge_dis_thre',
    'plane_detection_thre', 'voxel_size', 'voxel_init_num', 'proj_plane_num',
    'proj_image_resolution', 'proj_image_high_inc', 'proj_dis_min',
    'proj_dis_max', 'summary_min_thre', 'line_filter_enable',
    'descriptor_near_num', 'descriptor_min_len', 'descriptor_max_len',
    'non_max_suppression_radius', 'triangle_resolution', 'skip_near_num',
    'candidate_num', 'rough_dis_threshold', 'similarity_threshold',
    'icp_threshold', 'normal_threshold', 'dis_threshold',
    'geom_side_std_threshold', 'geom_center_std_threshold',
    'ransac_min_vote', 'ransac_max_iterations', 'ransac_sample_max',
    'ransac_correspondence_dis', 'candidate_selector_min_vote',
    'candidate_verify_min_pairs', 'icp_point_to_point_max',
]


def _write_btc_cpp_config(ros_params, source_file):
    """把 ROS2 参数格式中的 BTC 键转写为 OpenCV FileStorage 扁平格式临时文件"""
    import tempfile
    lines = ['%YAML:1.0',
             f'# auto-converted from {source_file} (ROS2 params -> OpenCV flat)']
    for k in _BTC_CPP_KEYS:
        if k in ros_params:
            lines.append(f"{k}: {ros_params[k]}")
    fd, tmp_path = tempfile.mkstemp(prefix='btc_cpp_flat_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return tmp_path


def _ros2_params_to_nested(p):
    """把 ROS2 扁平参数映射为 load_unified_config 期望的嵌套结构"""
    gicp = {}
    gicp_alias = {
        'fitness_score_threshold': 'gicp_fitness_score_threshold',
        'max_correspondence_distance': 'gicp_max_correspondence_distance',
        'max_iterations': 'gicp_max_iterations',
        'transformation_epsilon': 'gicp_transformation_epsilon',
        'scan_ds_size': 'gicp_scan_ds_size',
        'coarse_ds_size': 'gicp_coarse_ds_size',
        'coarse_max_iter': 'gicp_coarse_max_iter',
        'coarse_max_dist': 'gicp_coarse_max_dist',
        'max_init_translation': 'gicp_max_init_translation',
        'num_threads': 'gicp_num_threads',
    }
    for attr, ros_key in gicp_alias.items():
        if ros_key in p:
            gicp[attr] = p[ros_key]
    if 'use_gicp_for_loop_closure' in p:
        gicp['enabled'] = p['use_gicp_for_loop_closure']

    nested = {'gicp': gicp}
    if 'skip_near_num' in p:
        nested['skip_near_num'] = p['skip_near_num']

    sc = {k: p[k] for k in ('use_method', 'sc_dist_thres', 'sc_max_radius') if k in p}
    if sc:
        nested['scancontext'] = sc

    kf = {}
    if 'keyframe_meter_gap' in p:
        kf['meter_gap'] = p['keyframe_meter_gap']
    if 'keyframe_deg_gap' in p:
        kf['deg_gap'] = p['keyframe_deg_gap']
    if kf:
        nested['keyframe'] = kf

    lv = {}
    lv_flat = {
        'max_loop_distance': 'max_loop_distance',
        'max_yaw_diff': 'max_yaw_diff',
        'odom_direct_threshold': 'odom_direct_threshold',
        'loop_z_gate': 'z_gate',
        'loop_robust_kernel': 'robust_kernel',
        'loop_robust_k': 'robust_k',
        'enable_pcm': 'enable_pcm',
        'pcm_pos_thres': 'pcm_pos_thres',
        'pcm_rot_thres': 'pcm_rot_thres',
        'cross_floor_enabled': 'cross_floor_enabled',
        'cross_floor_max_dz': 'cross_floor_max_dz',
        'cross_floor_far_dist': 'cross_floor_far_dist',
        'cross_floor_min_pts': 'cross_floor_min_pts',
        'cross_floor_fitness_thres': 'cross_floor_fitness_thres',
    }
    for _flat_key, _attr in lv_flat.items():
        if _flat_key in p:
            lv[_attr] = p[_flat_key]
    if lv:
        nested['loop_validation'] = lv

    out = {k: p[k] for k in ('save_fused_scans',) if k in p}
    if out:
        nested['output'] = out

    return nested


def load_unified_config(config_file):
    """
    加载统一配置文件（btc + gicp + keyframe + loop_validation 合并为一个 yaml），
    返回 dict:
      - btc_config_path: str (C++ BTC 模块直接读原始路径)
      - gicp_config: GICPConfig
      - keyframe_meter_gap: float
      - keyframe_deg_gap: float
      - use_gicp: bool
      - scan_ds_size: float
      - max_loop_distance: float
      - max_yaw_diff: float

    兼容两种格式:
      - OpenCV FileStorage 扁平/嵌套 YAML（可含 %YAML:1.0 头）
      - ROS2 参数格式 (/**: ros__parameters: ...)：内部自动转换为
        Python 嵌套结构 + 临时 OpenCV 扁平文件供 C++ BTC 读取
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()
    # 跳过 OpenCV FileStorage 头 (%YAML:1.0)
    cleaned = ''.join(l for l in lines if not l.strip().startswith('%'))
    data = yaml.safe_load(cleaned)

    result = {}
    result['btc_config_path'] = config_file

    if _is_ros2_param_format(data):
        ros_params = (data['/**'] or {}).get('ros__parameters', {}) or {}
        result['btc_config_path'] = _write_btc_cpp_config(ros_params, config_file)
        data = _ros2_params_to_nested(ros_params)
        print(f"[Config] 检测到 ROS2 参数格式，已内部转换: {config_file}")

    # GICP 部分 — 从 YAML 读取所有参数，fallback 使用 GICPConfig 默认值（单一数据源）
    gicp_cfg = GICPConfig()
    g = data.get('gicp', {})
    for attr_name in ['fitness_score_threshold', 'max_correspondence_distance',
                      'max_iterations', 'transformation_epsilon', 'gicp_epsilon',
                      'scan_ds_size', 'coarse_ds_size', 'coarse_max_iter',
                      'coarse_max_dist', 'max_init_translation',
                      'rotation_epsilon', 'k_correspondences',
                      'max_optimizer_iterations', 'num_threads']:
        if attr_name in g:
            setattr(gicp_cfg, attr_name, g[attr_name])
    result['gicp_config'] = gicp_cfg

    # skip_near_num - 顶层键（在YAML中属于BTC检索部分，非嵌套）
    result['skip_near_num'] = data.get('skip_near_num', 5)  # 从配置文件读取

    # ScanContext部分 - 新增
    sc_cfg = data.get('scancontext', {})
    result['sc_dist_thres'] = sc_cfg.get('sc_dist_thres', 0.6)  # ScanContext距离阈值
    result['sc_max_radius'] = sc_cfg.get('sc_max_radius', 80.0)  # ScanContext最大半径
    result['use_method'] = sc_cfg.get('use_method', 'btc')  # 回环检测方法选择 ('btc' or 'sc')
    result['use_gicp'] = g.get('enabled', True)

    # 关键帧部分
    kf = data.get('keyframe', {})
    result['keyframe_meter_gap'] = kf.get('meter_gap', 1.0)
    result['keyframe_deg_gap'] = kf.get('deg_gap', 10.0)

    # 回环验证部分
    lv = data.get('loop_validation', {})
    result['max_loop_distance'] = lv.get('max_loop_distance', 100.0)
    result['max_yaw_diff'] = lv.get('max_yaw_diff', np.pi * 0.75)
    result['odom_direct_threshold'] = lv.get('odom_direct_threshold', 3.0)
    # 多楼层误回环治理
    result['loop_z_gate'] = lv.get('z_gate', 1.2)
    result['loop_robust_kernel'] = lv.get('robust_kernel', 'huber')
    result['loop_robust_k'] = lv.get('robust_k', 0.0)
    result['enable_pcm'] = lv.get('enable_pcm', False)
    result['pcm_pos_thres'] = lv.get('pcm_pos_thres', 1.5)
    result['pcm_rot_thres'] = lv.get('pcm_rot_thres', 0.2)
    # 跨楼层景观约束
    result['cross_floor_enabled'] = lv.get('cross_floor_enabled', False)
    result['cross_floor_max_dz'] = lv.get('cross_floor_max_dz', 4.0)
    result['cross_floor_far_dist'] = lv.get('cross_floor_far_dist', 20.0)
    result['cross_floor_min_pts'] = lv.get('cross_floor_min_pts', 100)
    result['cross_floor_fitness_thres'] = lv.get('cross_floor_fitness_thres', 0.5)

    # 输出部分
    out_cfg = data.get('output', {})
    # 是否输出回环匹配点云融合PCD到 FusedScans/（默认关闭，避免占用大量磁盘）
    result['save_fused_scans'] = out_cfg.get('save_fused_scans', False)

    return result


# ======================== 完整离线回环流程 (对应 C++ performSCLoopClosure + process_pg) ========================

class OfflineLoopCloser:
    """
    离线回环检测器，流程与 C++ 在线版本完全一致:
      1. 加载位姿和点云
      2. 关键帧选择 (keyframeMeterGap / keyframeRadGap)
      3. BTC 描述子生成 + 数据库构建 (BtcDescManager)
      4. 回环检测 (SearchLoop)
      5. GICP 精化 (可选)
      6. 回环验证 (validateLoopClosure)
      7. ISAM2 位姿图优化
    """

    def __init__(self, data_dir=None, btc_config_file=None, gicp_config=None,
                 keyframe_meter_gap=5.0, keyframe_deg_gap=10.0,
                 use_gicp=True, scan_ds_size=0.4, debug_btc=False,
                 max_loop_distance=100.0, max_yaw_diff=None,
                 odom_direct_threshold=3.0, skip_near_num=5,
                 use_method='btc', sc_dist_thres=0.6, sc_max_radius=80.0,
                 merge_n=1, num_threads=0, ros_node=None,
                 data_format='auto', scan_prefix='', save_fused_scans=False,
                 loop_z_gate=1.2, loop_robust_kernel='huber', loop_robust_k=0.0,
                 enable_pcm=False, pcm_pos_thres=1.5, pcm_rot_thres=0.2,
                 cross_floor_enabled=False, cross_floor_max_dz=4.0,
                 cross_floor_far_dist=20.0, cross_floor_min_pts=100,
                 cross_floor_fitness_thres=0.5):
        """
        离线回环检测器 - 支持BTC和ScanContext两种方法

        参数:
            debug_btc: bool, 默认False（开启C++ BTC详细调试日志）
            merge_n: int, 默认1（不合并）。每N帧合并为一帧提点云密度
            odom_direct_threshold: float, 默认3.0。odom距离小于此值时直接GICP验证
            skip_near_num: int, 默认5。跳过帧号差值<=此值的邻近帧
            use_method: str, 默认'btc'。回环检测方法 ('btc' or 'sc')
            sc_dist_thres: float, 默认0.6。ScanContext距离阈值
            sc_max_radius: float, 默认80.0。ScanContext最大半径
            num_threads: int, 默认0。多线程加速线程数 (0=自动检测CPU核心数)
            ros_node: rclpy.Node 或 None，若提供则发布 odom_path / optimized_path / loop_match_markers
                      话题用于 RViz 可视化比较 PGO 效果
            data_format: str, 默认'auto'。数据格式: 'auto' | 'kitti' | 'superlio'
                         - 'kitti':    odom_poses.txt (KITTI 12列) + Scans/000000.pcd
                         - 'superlio': Super-LIO 原生格式, 同目录下 scans_*.pcd + scans_*.txt 成对，
                                       pcd 为世界系累积点云(原点=程序起点)，txt 为
                                       "timestamp tx ty tz qx qy qz qw"
            scan_prefix: str, 默认''。superlio 格式的点云文件名前缀(多雷达时区分)
            save_fused_scans: bool, 默认False。是否输出回环匹配点云的融合PCD到
                             FusedScans/ 目录（仅供目视检查，点云翻倍输出较占磁盘）
        """
        self.data_format = data_format
        self.scan_prefix = scan_prefix
        self.save_fused_scans = save_fused_scans
        self.data_dir = data_dir or "/home/ywj/save_data/"
        self.debug_btc = debug_btc
        self.merge_n = max(1, int(merge_n))
        self.odom_direct_threshold = odom_direct_threshold
        self.skip_near_num = skip_near_num
        self.use_method = use_method  # 回环检测方法选择
        # 多线程参数: 0=自动检测CPU核心数
        self.num_threads = num_threads if num_threads > 0 else os.cpu_count() or 4
        self._btc_lock = threading.Lock()  # 保护C++ BTC数据库的线程安全（仅 BTC/SC 模式需要）
        self._ros_publish_lock = threading.Lock()  # 保护ROS2发布操作的线程安全（避免 rclpy.spin_once 冲突）

        self.btc_config_file = btc_config_file  # 缓存参数对比用

        # BTC C++ 模块
        if HAS_BTC_CPP and use_method == 'btc':
            import btc_cpp
            if btc_config_file and os.path.exists(btc_config_file):
                try:
                    self.btc_cpp = btc_cpp.BtcDescManager(btc_config_file)
                    print(f"[BTC] ✓ 使用C++实现 (加载配置): {btc_config_file}")
                except Exception as e:
                    self.btc_cpp = btc_cpp.BtcDescManager()
                    print(f"[BTC] ⚠ 配置文件无法被C++ BTC读取 ({e})，回退到内置默认配置")
            else:
                self.btc_cpp = btc_cpp.BtcDescManager()
                print("[BTC] ✓ 使用C++实现 (内置默认配置)")

            # 启用C++ debug日志
            if debug_btc:
                self.btc_cpp.SetDebugInfo(True)

            # 设置最大回环距离
            self.btc_cpp.SetMaxLoopDistance(max_loop_distance)
        else:
            self.btc_cpp = None
            if use_method == 'btc':
                print("[BTC] ⚠ C++ 模块未安装，无法使用BTC方法")

        # ScanContext C++ 模块
        if HAS_SC_CPP and use_method == 'sc':
            import sc_cpp
            self.sc_cpp = sc_cpp.SCManager()
            self.sc_cpp.setSCdistThres(sc_dist_thres)
            self.sc_cpp.setMaximumRadius(sc_max_radius)
            print(f"[ScanContext] ✓ 使用C++实现 (dist_thres={sc_dist_thres}, max_radius={sc_max_radius})")
        else:
            self.sc_cpp = None
            if use_method == 'sc':
                print("[ScanContext] ⚠ C++ 模块未安装，无法使用ScanContext方法")

        # odom_only 模式：不使用 BTC/SC，只根据 Odom 直接验证
        if use_method == 'odom_only':
            self.btc_cpp = None
            self.sc_cpp = None
            print("[Odom Only] ✓ 仅使用 Odom 距离进行 GICP 验证，不使用 BTC/SC")

        self.btc_manager = self.btc_cpp  # 统一接口（仅 BTC 模式使用）
        self.btc_config = None  # Python不再需要BTC配置对象

        # 回环验证参数
        self.max_loop_distance = max_loop_distance
        if max_yaw_diff is not None:
            self.max_yaw_diff = max_yaw_diff
        else:
            self.max_yaw_diff = np.pi * 0.75  # 默认 0.75π
            print("[BTC] Python BTC调试日志已开启")

        # 抗多楼层误回环 / 鲁棒后端参数
        self.loop_z_gate = float(loop_z_gate)          # 层高门控: |dz| 超此值视为跨楼层(<=0 关闭)
        self.loop_robust_kernel = loop_robust_kernel    # 回环因子鲁棒核: huber|cauchy|none
        self.loop_robust_k = float(loop_robust_k)       # 核尺度(<=0 用默认)
        self.enable_pcm = bool(enable_pcm)              # 是否启用回环一致性校验(PCM)
        self.pcm_pos_thres = float(pcm_pos_thres)       # PCM 平移一致性阈值(m)
        self.pcm_rot_thres = float(pcm_rot_thres)       # PCM 旋转一致性阈值(rad)
        # 跨楼层景观约束: 用窗外共同景观修正层间相对位姿(非同一层回环)
        self.cross_floor_enabled = bool(cross_floor_enabled)
        self.cross_floor_max_dz = float(cross_floor_max_dz)          # 允许的跨楼层 |dz| 上限(m)
        self.cross_floor_far_dist = float(cross_floor_far_dist)      # 远点距离判据(m), 窗外景观
        self.cross_floor_min_pts = int(cross_floor_min_pts)          # 每个点云最少远点数
        self.cross_floor_fitness_thres = float(cross_floor_fitness_thres)
        if self.loop_z_gate > 0:
            print(f"[LoopGuard] 层高门控: |dz| > {self.loop_z_gate}m 的候选按跨楼层剔除")
        if self.enable_pcm:
            print(f"[LoopGuard] PCM 一致性校验: 启用 (pos<{self.pcm_pos_thres}m, "
                  f"rot<{self.pcm_rot_thres}rad)")
        if self.cross_floor_enabled:
            print(f"[LoopGuard] 跨楼层景观约束: 启用 (dz∈({self.loop_z_gate}, "
                  f"{self.cross_floor_max_dz}]m, 远点>{self.cross_floor_far_dist}m "
                  f"且>={self.cross_floor_min_pts}个, fitness<{self.cross_floor_fitness_thres})")

        # GICP 配置
        self.gicp_config = gicp_config or GICPConfig()
        print(f"[GICP] 配置: fitness_thres={self.gicp_config.fitness_score_threshold:.2f}, "
              f"max_dist={self.gicp_config.max_correspondence_distance:.1f}, "
              f"max_iter={self.gicp_config.max_iterations}")
        self.use_gicp = use_gicp

        # 关键帧参数
        self.keyframe_meter_gap = keyframe_meter_gap
        self.keyframe_deg_gap = keyframe_deg_gap

        # 点云下采样
        self.scan_ds_size = scan_ds_size

        # 数据存储
        self.keyframe_clouds = []       # 原始关键帧点云 (BTC 使用)
        self.keyframe_clouds_ds = []    # 下采样关键帧点云 (GICP 使用)
        self.keyframe_poses = []     # 关键帧位姿 4x4
        self.keyframe_poses6d = []   # 关键帧位姿 (x,y,z,roll,pitch,yaw)
        self.keyframe_times = []

        # 多帧合并映射 (merge_n > 1 时使用)
        self.merge_indices = []      # merged_idx -> [原始帧索引列表]
        self.original_poses = None   # 保留原始所有帧位姿（合并前备份）
        self.original_scans = None   # 保留原始所有帧点云（合并前备份）

        # 回环结果
        self.loop_pairs = []         # [(frame_a, frame_b, score, fitness), ...]
        self.loop_details = []       # 详细信息

        # ===== ROS2 可视化（可选）=====
        # 话题命名与在线 C++ 版本一致：odom_keyframe_path / optimized_path / loop_match_markers
        self.ros_node = ros_node
        self._ros_loop_marker_id = 0  # 递增的 marker id
        self._ros_pub_odom_path = None
        self._ros_pub_optimized_path = None
        self._ros_pub_loop_markers = None
        if ros_node is not None:
            if not HAS_ROS2:
                print("[WARN] rclpy 不可用，ros_node 参数将被忽略")
                self.ros_node = None
            else:
                # TRANSIENT_LOCAL 让晚加入的订阅者也能收到最新数据
                qos = rclpy.qos.QoSProfile(depth=10,
                                           durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
                self._ros_pub_odom_path = ros_node.create_publisher(
                    RosPath, 'odom_keyframe_path', qos)
                self._ros_pub_optimized_path = ros_node.create_publisher(
                    RosPath, 'optimized_path', qos)
                self._ros_pub_loop_markers = ros_node.create_publisher(
                    RosMarkerArray, 'loop_match_markers', qos)
                self._ros_pub_current_odom = ros_node.create_publisher(
                    RosOdometry, 'loop_closure_progress', 10)
                print("[ROS2] 离线回环可视化已启用: odom_keyframe_path / optimized_path / loop_match_markers")
                print("[ROS2]   进度跟踪: loop_closure_progress (Odometry)")

    # ======================== BTC 缓存 ========================

    def _btc_cache_dir(self):
        """BTC 缓存目录"""
        return os.path.join(self.data_dir, "BTC_CACHE")

    def _btc_cache_params(self):
        """收集影响 BTC 生成的所有参数，用于判断缓存是否有效"""
        params = {
            'merge_n': self.merge_n,
            'keyframe_meter_gap': self.keyframe_meter_gap,
            'keyframe_deg_gap': self.keyframe_deg_gap,
            'scan_ds_size': self.scan_ds_size,
        }
        if self.btc_config_file and os.path.exists(self.btc_config_file):
            try:
                with open(self.btc_config_file, 'r') as f:
                    params['btc_config_content'] = f.read()
            except Exception:
                pass
        pairs = self._find_superlio_pairs(self.data_dir)
        if pairs:
            params['poses_mtime'] = max(os.path.getmtime(p[1]) for p in pairs)
        elif os.path.exists(os.path.join(self.data_dir, "odom_poses.txt")):
            params['poses_mtime'] = os.path.getmtime(
                os.path.join(self.data_dir, "odom_poses.txt"))
        return params

    def _btc_cache_params_match(self):
        """与缓存的参数对比，返回 True 表示匹配"""
        import pickle, json
        params_file = os.path.join(self._btc_cache_dir(), "params.json")
        if not os.path.exists(params_file):
            return False
        try:
            with open(params_file, 'r') as f:
                saved = json.load(f)
        except Exception:
            return False

        current = self._btc_cache_params()
        # 把 btc_config_content 从 params 中排除（JSON中不存大文本）
        # 其他关键参数逐项对比
        for key in ['merge_n', 'keyframe_meter_gap', 'keyframe_deg_gap', 'scan_ds_size']:
            if saved.get(key) != current.get(key):
                print(f"[Cache] 参数不匹配 ({key}): saved={saved.get(key)}, current={current.get(key)}")
                return False
        if saved.get('poses_mtime') != current.get('poses_mtime'):
            print("[Cache] 位姿文件已更新")
            return False
        return True

    def _btc_cache_save_params(self):
        """保存当前参数到 params.json"""
        import json
        cache_dir = self._btc_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        params = self._btc_cache_params()
        # 不保存 btc_config_content（可能很大）
        params.pop('btc_config_content', None)
        with open(os.path.join(cache_dir, "params.json"), 'w') as f:
            json.dump(params, f)

    def _btc_cache_frame_path(self, frame_idx):
        """单帧BTC缓存文件路径"""
        return os.path.join(self._btc_cache_dir(), f"{frame_idx:06d}.btc")

    def _btc_cache_clear(self):
        """清除旧的缓存文件"""
        import shutil
        cache_dir = self._btc_cache_dir()
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    def _try_load_btc_cache(self, total_frames):
        """尝试从缓存加载 BTC 描述子。
        返回 (loaded_set: set[int], reason: str)。
        loaded_set 中已完成的帧索引，未完成的帧需要重新生成。
        """
        if self.btc_manager is None:
            return set(), "btc_manager is None"

        import pickle
        cache_dir = self._btc_cache_dir()

        # 检查参数是否匹配
        if not self._btc_cache_params_match():
            # 参数变了，清除全部旧缓存
            self._btc_cache_clear()
            self._btc_cache_save_params()
            return set(), "参数已变更，清除旧缓存"

        # 参数匹配，扫描已有的帧缓存
        loaded = set()
        total_btcs = 0
        for i in range(total_frames):
            fpath = self._btc_cache_frame_path(i)
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'rb') as f:
                        item = pickle.load(f)
                    btcs = item.get('btcs_data', [])
                    position = np.array(item.get('frame_position', [0, 0, 0]), dtype=np.float64)
                    if btcs:
                        self.btc_manager.AddBtcDescsFromCache(btcs, position)
                        total_btcs += len(btcs)
                    loaded.add(i)
                except Exception as e:
                    print(f"[Cache] 帧 {i} 缓存损坏: {e}, 将重新生成")

        missing = total_frames - len(loaded)
        if missing == 0:
            print(f"[Cache] ✓ 全部 {len(loaded)} 帧已加载 ({total_btcs} 个 BTC)")
        elif len(loaded) > 0:
            print(f"[Cache] 已加载 {len(loaded)}/{total_frames} 帧 ({total_btcs} 个 BTC), "
                  f"剩余 {missing} 帧需要生成")
        else:
            print(f"[Cache] 无有效缓存, {total_frames} 帧需要生成")

        return loaded, ""

    def load_data(self, data_dir=None):
        """加载位姿和点云数据（自动识别 KITTI 或 Super-LIO 原生格式）"""
        data_dir = data_dir or self.data_dir

        fmt = self._detect_data_format(data_dir)
        if fmt == 'kitti':
            return self._load_data_kitti(data_dir)
        elif fmt == 'superlio':
            return self._load_data_superlio(data_dir)
        else:
            print(f"[ERROR] 无法识别数据格式: {data_dir}")
            print("        KITTI:    需要 odom_poses.txt (12列) + Scans/000000.pcd")
            print("        Super-LIO: 需要成对的 scans_*.pcd + scans_*.txt")
            return False

    def _detect_data_format(self, data_dir):
        """识别数据格式: 'kitti' | 'superlio' | None"""
        if self.data_format and self.data_format != 'auto':
            return self.data_format
        if os.path.exists(os.path.join(data_dir, "odom_poses.txt")) and \
           os.path.isdir(os.path.join(data_dir, "Scans")):
            return 'kitti'
        if self._find_superlio_pairs(data_dir):
            return 'superlio'
        return None

    def _load_data_kitti(self, data_dir):
        """加载 KITTI 格式: odom_poses.txt + Scans/000000.pcd"""
        poses_file = os.path.join(data_dir, "odom_poses.txt")
        if not os.path.exists(poses_file):
            print(f"[ERROR] 位姿文件不存在: {poses_file}")
            return False

        self.all_poses = self._load_poses_kitti(poses_file)
        print(f"[Load] 加载了 {len(self.all_poses)} 个位姿")

        # 如果合并缓存命中，跳过原始点云加载（_merge_frames 会直接从缓存加载）
        if self.merge_n > 1 and self._merge_cache_hit():
            self.all_scans = []  # _merge_frames 会填充
            return True

        scans_dir = os.path.join(data_dir, "Scans")
        if not os.path.isdir(scans_dir):
            print(f"[ERROR] 点云目录不存在: {scans_dir}")
            return False

        self.all_scans = []
        for i in range(len(self.all_poses)):
            pcd_file = os.path.join(scans_dir, f"{i:06d}.pcd")
            if os.path.exists(pcd_file):
                pts = self._load_pcd(pcd_file)
                self.all_scans.append(pts)
            else:
                print(f"[WARN] 点云文件缺失: {pcd_file}")
                self.all_scans.append(np.empty((0, 3)))

        print(f"[Load] 加载了 {len(self.all_scans)} 帧点云")
        return True

    @staticmethod
    def _trailing_index(filename):
        """提取文件名末尾数字，用于按帧号排序；无数字返回 None"""
        stem = os.path.splitext(filename)[0]
        m = re.search(r'(\d+)$', stem)
        return int(m.group(1)) if m else None

    def _find_superlio_pairs(self, data_dir):
        """查找 Super-LIO 原生格式的 scans_*.pcd + scans_*.txt 成对文件"""
        if not os.path.isdir(data_dir):
            return []
        pairs = []
        for fn in os.listdir(data_dir):
            if not fn.endswith('.pcd'):
                continue
            if self.scan_prefix and not fn.startswith(self.scan_prefix):
                continue
            txt_fn = fn[:-4] + '.txt'
            txt_path = os.path.join(data_dir, txt_fn)
            if not os.path.exists(txt_path):
                continue
            idx = self._trailing_index(fn)
            pairs.append((idx, txt_path, os.path.join(data_dir, fn)))
        # 优先按帧号排序，无帧号的排在最后按文件名排序
        pairs.sort(key=lambda p: (p[0] is None, p[0] if p[0] is not None else 0, p[1]))
        return pairs

    def _load_data_superlio(self, data_dir):
        """
        加载 Super-LIO 原生格式。

        pcd 为世界系(地图系)累积点云，所有帧共享同一原点(程序起点)；
        txt 每行: timestamp tx ty tz qx qy qz qw (位姿为 body→world)。

        为了让 SC-PGO 流程（BTC/GICP 均假设输入为传感器系局部点云）正常工作，
        这里将世界系点云转换到对应帧的 body 系:
            p_body = R^T * (p_world - t)
        """
        pairs = self._find_superlio_pairs(data_dir)
        if not pairs:
            print(f"[ERROR] 未找到 scans_*.pcd + scans_*.txt 成对文件: {data_dir}")
            return False

        print(f"[Load] 检测到 Super-LIO 原生格式: {len(pairs)} 帧 (world→body 转换)")

        if self.merge_n > 1 and self._merge_cache_hit():
            self.all_poses = self._load_poses_kitti(
                os.path.join(data_dir, "merged_odom_poses.txt"))
            self.all_scans = []
            return True

        self.all_poses = []
        self.all_scans = []
        bad_pose = 0
        _t_parse = 0.0        # 位姿 txt 解析
        _t_read = 0.0         # pcd 读取（含二进制解析）
        _t_xform = 0.0        # world→body 变换
        _pts_total = 0
        _read_max_dt = 0.0    # 最慢单帧 pcd 读取耗时
        _read_max_pts = 0
        for idx, txt_path, pcd_path in pairs:
            _t0 = time.perf_counter()
            T, ts = self._load_superlio_pose(txt_path)
            _t_parse += time.perf_counter() - _t0
            if T is None:
                print(f"[WARN] 位姿文件无法解析: {txt_path}")
                bad_pose += 1
                continue

            _t0 = time.perf_counter()
            pts_world = self._load_pcd(pcd_path)
            _dt_read = time.perf_counter() - _t0
            _t_read += _dt_read
            if _dt_read > _read_max_dt:
                _read_max_dt = _dt_read
                _read_max_pts = len(pts_world)
            if len(pts_world) == 0:
                print(f"[WARN] 点云文件为空: {pcd_path}")
                self.all_poses.append(T)
                self.all_scans.append(np.empty((0, 3)))
                continue

            # world → body: p_body = R^T (p_world - t)
            _t0 = time.perf_counter()
            R = T[:3, :3]
            t = T[:3, 3]
            pts_body = (R.T @ (pts_world - t).T).T
            _t_xform += time.perf_counter() - _t0

            _pts_total += len(pts_body)
            self.all_poses.append(T)
            self.all_scans.append(np.ascontiguousarray(pts_body, dtype=np.float32))

        if bad_pose:
            print(f"[WARN] 跳过 {bad_pose} 个无效位姿")
        print(f"[Load] 加载了 {len(self.all_poses)} 个位姿, {len(self.all_scans)} 帧点云")
        print(f"  [Load-Timing] 位姿解析 {_t_parse:.2f}s | PCD读取 {_t_read:.2f}s | "
              f"world→body {_t_xform:.2f}s | 总点数 {_pts_total:,} "
              f"(均 {_pts_total / max(len(self.all_scans), 1):.0f})")
        print(f"  [Load-Timing] 最慢单帧 PCD读取 {_read_max_dt:.2f}s ({_read_max_pts:,} 点)")
        return len(self.all_poses) > 0

    @staticmethod
    def _load_superlio_pose(filepath):
        """
        解析 Super-LIO 原生位姿文件，返回 (4x4矩阵, timestamp)。
        格式: timestamp tx ty tz qx qy qz qw
        """
        try:
            with open(filepath, 'r') as f:
                line = f.readline().strip()
            if not line:
                return None, None
            vals = [float(v) for v in line.split()]
            if len(vals) < 8:
                return None, None
            ts = vals[0]
            t = np.array(vals[1:4], dtype=np.float64)
            R = _quat_xyzw_to_matrix(vals[4], vals[5], vals[6], vals[7])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            return T, ts
        except Exception as e:
            print(f"[ERROR] 解析位姿失败: {filepath}, {e}")
            return None, None

    def _merge_cache_hit(self):
        """检查合并缓存是否存在且数量匹配"""
        n = self.merge_n
        total = len(self.all_poses)
        num_groups = (total + n - 1) // n
        merged_dir = os.path.join(self.data_dir, "MergedScans")
        merged_poses_path = os.path.join(self.data_dir, "merged_odom_poses.txt")
        if not os.path.isdir(merged_dir) or not os.path.exists(merged_poses_path):
            return False
        existing_pcds = sorted([f for f in os.listdir(merged_dir) if f.endswith('.pcd')])
        cached_poses = self._load_poses_kitti(merged_poses_path)
        return len(existing_pcds) == num_groups and len(cached_poses) == num_groups

    def _merge_frames(self):
        """
        将每 merge_n 帧合并为一帧（针对点云稀疏雷达如 Mid360）。

        利用 odom_poses 将组内所有帧的点云变换到该组最后一帧的坐标系下，
        合并为一帧更稠密的点云。合并后的帧位姿取组内最后一帧的位姿。
        """
        if self.merge_n <= 1:
            return  # 不需要合并

        n = self.merge_n
        total = len(self.all_poses)
        num_groups = (total + n - 1) // n

        print(f"\n===== 多帧合并 (每{n}帧→1帧) =====")
        print(f"  原始帧数: {total}, 合并组数: {num_groups}")

        # 检查是否已有缓存的合并结果
        merged_dir = os.path.join(self.data_dir, "MergedScans")
        merged_poses_path = os.path.join(self.data_dir, "merged_odom_poses.txt")
        if os.path.isdir(merged_dir) and os.path.exists(merged_poses_path):
            # 统计 MergedScans 下的 pcd 数量
            existing_pcds = sorted([
                f for f in os.listdir(merged_dir) if f.endswith('.pcd')
            ])
            cached_poses = self._load_poses_kitti(merged_poses_path)
            if len(existing_pcds) == num_groups and len(cached_poses) == num_groups:
                print(f"[Cache] 合并结果缓存命中: {len(existing_pcds)} 帧, 直接加载")
                self.original_poses = [T.copy() for T in self.all_poses]
                self.original_scans = list(self.all_scans)
                self.merge_indices = [
                    list(range(g * n, min((g + 1) * n, total)))
                    for g in range(num_groups)
                ]
                self.all_poses = cached_poses
                self.all_scans = []
                # 固定输出 10 次进度，均匀分布
                progress_interval = max(1, num_groups // 10)
                progress_indices = set(range(0, num_groups, progress_interval))
                # 确保包含最后一帧
                if num_groups > 1:
                    progress_indices.add(num_groups - 1)
                for g in range(num_groups):
                    pcd_path = os.path.join(merged_dir, existing_pcds[g])
                    pts = self._load_pcd(pcd_path)
                    self.all_scans.append(pts)
                    if g in progress_indices:
                        print(f"  加载合并帧 {g:4d}/{num_groups}: {len(pts)} pts")
                return
            else:
                print(f"[Cache] 合并结果数量不匹配 (缓存={len(existing_pcds)}/{len(cached_poses)}, "
                      f"需要={num_groups}), 重新合并")

        # 备份原始数据
        self.original_poses = [T.copy() for T in self.all_poses]
        self.original_scans = list(self.all_scans)

        merged_poses = []
        merged_scans = []
        self.merge_indices = []

        for g in range(num_groups):
            start = g * n
            end = min(start + n, total)
            group_indices = list(range(start, end))
            ref_idx = end - 1  # 以组内最后一帧为参考坐标系
            T_ref = self.all_poses[ref_idx]
            T_ref_inv = np.linalg.inv(T_ref)

            all_pts_world = []

            for idx in group_indices:
                pts = self.all_scans[idx]
                if len(pts) == 0:
                    continue
                T_i = self.all_poses[idx]
                # 将帧i的点云变换到参考帧坐标系: point_ref = inv(T_ref) * T_i * point_i
                T_rel = T_ref_inv @ T_i
                R_rel = T_rel[:3, :3]
                t_rel = T_rel[:3, 3]
                pts_trans = (R_rel @ pts.T).T + t_rel
                all_pts_world.append(pts_trans)

            if len(all_pts_world) > 0:
                merged_cloud = np.vstack(all_pts_world)
            else:
                merged_cloud = np.empty((0, 3))

            merged_poses.append(T_ref.copy())
            merged_scans.append(merged_cloud)
            self.merge_indices.append(group_indices)

            # 统计
            total_pts = sum(len(self.all_scans[idx]) for idx in group_indices)
            if g < 5 or g >= num_groups - 3:
                print(f"  组 {g:4d}: 帧 [{start}, {end-1}], 参考帧={ref_idx}, "
                      f"输入 {total_pts} pts -> 合并 {len(merged_cloud)} pts")

        # 替换 all_poses 和 all_scans 为合并后的数据
        self.all_poses = merged_poses
        self.all_scans = merged_scans
        print(f"  合并完成: {len(merged_poses)} 帧 (每帧约 {n} 倍点数)")

        # 保存合并后的点云到 MergedScans/ 目录（局部坐标系，组内最后一帧坐标系）
        merged_dir = os.path.join(self.data_dir, "MergedScans")
        os.makedirs(merged_dir, exist_ok=True)
        print(f"  保存合并点云到: {merged_dir}")
        for g, cloud in enumerate(merged_scans):
            if len(cloud) > 0:
                out_path = os.path.join(merged_dir, f"{g:06d}.pcd")
                _save_pcd_numpy(out_path, cloud)

        # 另存一份 world 坐标系版本（用于手动检查回环位置）
        merged_world_dir = os.path.join(self.data_dir, "MergedScansWorld")
        os.makedirs(merged_world_dir, exist_ok=True)
        print(f"  保存 world 坐标系合并点云到: {merged_world_dir}")
        for g in range(num_groups):
            start = g * n
            end = min(start + n, total)
            group_indices = list(range(start, end))

            all_pts_world = []
            for idx in group_indices:
                pts = self.original_scans[idx]
                if len(pts) == 0:
                    continue
                T_i = self.original_poses[idx]
                R_i = T_i[:3, :3]
                t_i = T_i[:3, 3]
                pts_world = (R_i @ pts.T).T + t_i
                all_pts_world.append(pts_world)

            if len(all_pts_world) > 0:
                merged_world = np.vstack(all_pts_world)
                out_path = os.path.join(merged_world_dir, f"{g:06d}.pcd")
                _save_pcd_numpy(out_path, merged_world)

        # 也保存合并后的 odom 位姿（方便对应查看）
        merged_poses_path = os.path.join(self.data_dir, "merged_odom_poses.txt")
        self._save_poses_kitti(merged_poses_path, merged_poses)
        print(f"  保存合并位姿到: {merged_poses_path}")

    # ======================== ROS2 可视化辅助方法 ========================

    @staticmethod
    def _matrix_to_pose_stamped(T, frame_id, stamp_ns=0):
        """4x4 变换矩阵 → geometry_msgs/PoseStamped"""
        roll, pitch, yaw = _extract_rpy(T)
        t = T[:3, 3]
        ps = RosPoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = RosTime()  # zero stamp placeholder
        # rclpy time 内部用 int64 nanoseconds，这里直接用 stamp_ns
        ps.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        ps.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        # RPY -> quaternion (xyzw)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        ps.pose.orientation.w = float(cr * cp * cy + sr * sp * sy)
        ps.pose.orientation.x = float(sr * cp * cy - cr * sp * sy)
        ps.pose.orientation.y = float(cr * sp * cy + sr * cp * sy)
        ps.pose.orientation.z = float(cr * cp * sy - sr * sp * cy)
        return ps

    def _ros_publish_path(self, poses_list, publisher, frame_id='odom'):
        """发布 nav_msgs/Path（一整条轨迹）"""
        if self.ros_node is None or not HAS_ROS2 or publisher is None:
            return
        msg = RosPath()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        for T in poses_list:
            msg.poses.append(self._matrix_to_pose_stamped(T, frame_id))
        publisher.publish(msg)
        # 强制刷新确保消息发到wire（用锁避免多线程 rclpy.spin_once 冲突）
        import rclpy
        with self._ros_publish_lock:
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

    def _ros_publish_loop_markers(self, prev_idx, curr_idx, score,
                                  pose_prev, pose_curr, frame_id='odom'):
        """
        发布一对回环匹配点 + 连线，与在线 C++ 版 publishLoopMatchMarkers 格式一致。

        pose_prev, pose_curr: 4x4 numpy 矩阵
        """
        if self.ros_node is None or not HAS_ROS2:
            return
        now = self.ros_node.get_clock().now().to_msg()
        marker_array = RosMarkerArray()
        base_id = self._ros_loop_marker_id
        self._ros_loop_marker_id += 4  # 4 markers per loop event

        p_prev = pose_prev[:3, 3]
        p_curr = pose_curr[:3, 3]

        def _make_sphere(point, color_rgba, marker_id):
            m = RosMarker()
            m.header.frame_id = frame_id
            m.header.stamp = now
            m.ns = 'loop_match_points'
            m.id = marker_id
            m.type = RosMarker.SPHERE
            m.action = RosMarker.ADD
            m.pose.position.x = float(point[0])
            m.pose.position.y = float(point[1])
            m.pose.position.z = float(point[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.6
            m.color.r, m.color.g, m.color.b, m.color.a = color_rgba
            # lifetime = 0 表示永久
            m.lifetime = RosDuration()
            return m

        # Marker 1: prev keyframe point (red sphere)
        marker_array.markers.append(
            _make_sphere(p_prev, (1.0, 0.0, 0.0, 1.0), base_id))
        # Marker 2: curr keyframe point (green sphere)
        marker_array.markers.append(
            _make_sphere(p_curr, (0.0, 1.0, 0.0, 1.0), base_id + 1))

        # Marker 3: connecting line (cyan LINE_LIST with 2 points)
        m_line = RosMarker()
        m_line.header.frame_id = frame_id
        m_line.header.stamp = now
        m_line.ns = 'loop_match_lines'
        m_line.id = base_id
        m_line.type = RosMarker.LINE_LIST
        m_line.action = RosMarker.ADD
        m_line.scale.x = 0.05  # line width
        m_line.color.r = 0.0
        m_line.color.g = 1.0
        m_line.color.b = 1.0
        m_line.color.a = 1.0
        m_line.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point as RosPoint
        pt_prev = RosPoint(x=float(p_prev[0]), y=float(p_prev[1]), z=float(p_prev[2]))
        pt_curr = RosPoint(x=float(p_curr[0]), y=float(p_curr[1]), z=float(p_curr[2]))
        m_line.points = [pt_prev, pt_curr]
        m_line.lifetime = RosDuration()
        marker_array.markers.append(m_line)

        # Marker 4: text label (loop pair indices + score)
        m_text = RosMarker()
        m_text.header.frame_id = frame_id
        m_text.header.stamp = now
        m_text.ns = 'loop_match_labels'
        m_text.id = base_id
        m_text.type = RosMarker.TEXT_VIEW_FACING
        m_text.action = RosMarker.ADD
        m_text.pose.position.x = float((p_prev[0] + p_curr[0]) * 0.5)
        m_text.pose.position.y = float((p_prev[1] + p_curr[1]) * 0.5)
        m_text.pose.position.z = float((p_prev[2] + p_curr[2]) * 0.5 + 1.0)
        m_text.pose.orientation.w = 1.0
        m_text.scale.z = 0.5
        m_text.color.r = m_text.color.g = m_text.color.b = 1.0
        m_text.color.a = 1.0
        m_text.text = f'loop {prev_idx}<->{curr_idx} s={score:.3f}'
        m_text.lifetime = RosDuration()
        marker_array.markers.append(m_text)

        self._ros_pub_loop_markers.publish(marker_array)
        # 强制刷新确保消息发到wire（rclpy publish() 是异步入队，需spin_once触发发送）
        # 用锁避免多线程 rclpy.spin_once 冲突
        import rclpy
        with self._ros_publish_lock:
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

    def _ros_publish_current_odom(self, frame_idx, total_frames):
        """
        发布当前处理帧的 odometry 到 loop_closure_progress 话题。

        frame_idx: int, 当前帧索引
        total_frames: int, 关键帧总数
        """
        if self.ros_node is None or not HAS_ROS2 or self._ros_pub_current_odom is None:
            return
        T = self.keyframe_poses[frame_idx]
        odom = RosOdometry()
        odom.header.frame_id = 'odom'
        odom.header.stamp = self.ros_node.get_clock().now().to_msg()
        # 利用 child_frame_id 传递进度信息，方便 RViz 显示
        odom.child_frame_id = f'{frame_idx}/{total_frames}'

        t = T[:3, 3]
        odom.pose.pose.position.x = float(t[0])
        odom.pose.pose.position.y = float(t[1])
        odom.pose.pose.position.z = float(t[2])

        # 4x4 -> quaternion
        R = T[:3, :3]
        qw = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
        qx = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
        qy = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
        qz = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2
        # 取符号
        if R[2,1] - R[1,2] < 0: qx = -qx
        if R[0,2] - R[2,0] < 0: qy = -qy
        if R[1,0] - R[0,1] < 0: qz = -qz

        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        # 速度场留空，pose covariance 保留默认（全零）
        self._ros_pub_current_odom.publish(odom)
        import rclpy
        with self._ros_publish_lock:
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

    def run(self):
        """执行完整的离线回环流程"""
        if not hasattr(self, 'all_poses'):
            print("[ERROR] 请先调用 load_data()")
            return

        # ===== 性能计时 =====
        GICP_STATS.reset()
        _t_run0 = time.perf_counter()
        _t_prev = _t_run0
        _timings = []

        def _lap(name):
            nonlocal _t_prev
            _now = time.perf_counter()
            _timings.append((name, _now - _t_prev))
            _t_prev = _now
            print(f"  [Timing] {name}: {_timings[-1][1]:.2f}s")

        def _timing_report():
            total = time.perf_counter() - _t_run0
            print("\n========== 耗时汇总 ==========")
            for name, dt in _timings:
                print(f"  {name:<22}: {dt:>8.2f}s  ({dt / total * 100:>5.1f}%)")
            print(f"  {'总计':<22}: {total:>8.2f}s")
            GICP_STATS.report()

        # ===== 线程超订保护: 外层线程 × GICP内层线程 不应超过 CPU 核数 =====
        if self.use_gicp and HAS_GICP_OMP:
            _cores = os.cpu_count() or 1
            _gicp_threads = max(1, int(getattr(self.gicp_config, 'num_threads', 1)))
            _max_outer = max(1, _cores // _gicp_threads)
            if self.num_threads > _max_outer:
                print(f"[WARN] 线程超订: 外层 {self.num_threads} × GICP内层 {_gicp_threads} = "
                      f"{self.num_threads * _gicp_threads} > CPU {_cores} 核，"
                      f"自动将外层线程降为 {_max_outer}")
                self.num_threads = _max_outer

        # ===== 步骤 0: 多帧合并 (可选) =====
        self._merge_frames()
        _lap("步骤0 多帧合并")

        # ===== 步骤 1: 关键帧选择 (对应 C++ process_pg 中的关键帧判断) =====
        print("\n===== 步骤 1: 关键帧选择 =====")
        print(f"  参数: meter_gap={self.keyframe_meter_gap:.1f}m, "
              f"deg_gap={self.keyframe_deg_gap:.1f}°")
        self._select_keyframes()
        print(f"  关键帧数: {len(self.keyframe_poses)} / {len(self.all_poses)}")
        _lap("步骤1 关键帧选择")

        if len(self.keyframe_poses) < 20:
            print("[WARN] 关键帧数不足 20，无法进行回环检测")
            return

        # 发布原始 odom 关键帧轨迹（PGO 输入），用于在 RViz 中与优化结果对比
        self._ros_publish_path(self.keyframe_poses, self._ros_pub_odom_path)

        # ===== 预计算关键帧距离矩阵（优化 Step 3 Odom扫描）=====
        print("\n===== 预计算距离矩阵 =====")
        num_kf = len(self.keyframe_poses)
        positions = np.array([pose[:3, 3] for pose in self.keyframe_poses])  # N x 3
        # 完全向量化计算距离矩阵：利用广播计算所有帧对距离
        # positions[:, None, :] - positions[None, :, :] -> N x N x 3
        diff = positions[:, None, :] - positions[None, :, :]
        self.keyframe_distance_matrix = np.linalg.norm(diff, axis=2).astype(np.float32)
        print(f"  距离矩阵 {num_kf}x{num_kf} 已预计算完成")
        _lap("距离矩阵预计算")

        # ===== 步骤 2: BTC 描述子生成 + 数据库构建 =====
        print("\n===== 步骤 2: BTC 描述子生成 =====")
        step2_total = len(self.keyframe_clouds)

        if self.btc_manager is None:
            # odom_only / sc 模式：不需要 BTC 描述子
            print("  [Skip] 当前模式 ({} ) 不需要生成 BTC 描述子".format(self.use_method))
            loaded_set = set()
            to_generate = []
        else:
            # 尝试从缓存加载已有帧
            loaded_set, reason = self._try_load_btc_cache(step2_total)
            if reason:
                print(f"  [Cache] {reason}")

            # 确保 params.json 存在
            cache_dir = self._btc_cache_dir()
            params_file = os.path.join(cache_dir, "params.json")
            if not os.path.exists(params_file):
                self._btc_cache_save_params()

            # 确定需要生成的帧
            to_generate = [i for i in range(step2_total) if i not in loaded_set]
        already_done = len(loaded_set)

        if not to_generate:
            if loaded_set:
                print(f"  全部 {step2_total} 帧已缓存，跳过生成")
        else:
            print(f"  需要生成: {len(to_generate)} 帧, 已缓存: {already_done} 帧")
            print(f"  线程数: {self.num_threads}")

            import pickle
            zero_btc_frames = 0
            done_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._process_frame_step2, i): i
                    for i in to_generate
                }

                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        frame_result = future.result()
                    except Exception as e:
                        print(f"  [ERROR] 关键帧 {i} BTC生成异常: {e}")
                        continue

                    if frame_result is None:
                        continue

                    # 生成立即写入单帧缓存文件
                    if frame_result.get('btcs_data'):
                        fpath = self._btc_cache_frame_path(i)
                        with open(fpath, 'wb') as f:
                            save_item = {
                                'btcs_data': frame_result['btcs_data'],
                                'frame_position': frame_result['frame_position'],
                            }
                            pickle.dump(save_item, f, protocol=pickle.HIGHEST_PROTOCOL)

                    done_count += 1
                    n_btcs = frame_result['num_btcs']

                    if n_btcs == 0:
                        zero_btc_frames += 1

                    if not frame_result['success']:
                        if zero_btc_frames <= 5 or done_count >= len(to_generate):
                            print(f"  [WARN] 关键帧 {i}/{step2_total}: {frame_result.get('error', '失败')}")
                    elif n_btcs == 0:
                        if zero_btc_frames <= 5 or done_count >= len(to_generate):
                            print(f"  [WARN] 关键帧 {i}/{step2_total} 点数={frame_result['n_pts']}, BTC=0")
                    elif done_count % 20 == 0 or done_count >= len(to_generate) or done_count <= 5:
                        print(f"  关键帧 {i}/{step2_total} | 点云: {frame_result['n_pts']} pts | "
                              f"BTC: {n_btcs} | 进度: {done_count}/{len(to_generate)}")

        db_size = self.btc_manager.GetDatabaseSize() if hasattr(self.btc_manager, 'GetDatabaseSize') else 'N/A'
        print(f"\n  数据库大小: {db_size}, 总帧: {step2_total}")
        _lap("步骤2 BTC描述子")

        # ===== 步骤 3: 回环检测 + GICP 精化 + 验证 =====
        print("\n===== 步骤 3: 回环检测 (多线程加速) =====")
        print(f"  线程数: {self.num_threads}")

        # ----- 性能诊断: odom候选对数量 / 线程超订 -----
        _thr = self.odom_direct_threshold
        _skip = self.skip_near_num
        _zgate = getattr(self, 'loop_z_gate', 0.0)
        _xf_on = getattr(self, 'cross_floor_enabled', False)
        _xf_max_dz = getattr(self, 'cross_floor_max_dz', 4.0)
        _cand = 0
        _cand_dz = 0       # 被层高门控剔除(且未进入跨楼层通道)的数量
        _cand_xf = 0       # 进入跨楼层景观约束通道的数量
        for _i in range(len(self.keyframe_poses)):
            # 与 _process_frame_step3 一致: frame_diff > skip_near 且距离 < 阈值
            # distances[j] 对应 j=0..i-1, frame_diff = i-j > skip -> j < i-skip
            if _i <= _skip:
                continue
            _d = self.keyframe_distance_matrix[_i, :_i - _skip]
            _z_i = self.keyframe_poses[_i][2, 3]
            for _j in np.nonzero(_d < _thr)[0]:
                _dz = abs(_z_i - self.keyframe_poses[_j][2, 3])
                if _zgate > 0 and _dz > _zgate:
                    if _xf_on and _dz <= _xf_max_dz:
                        _cand_xf += 1
                    else:
                        _cand_dz += 1
                else:
                    _cand += 1
        _gicp_threads = getattr(self.gicp_config, 'num_threads', 1)
        _cores = os.cpu_count() or 1
        print(f"  [Diag] odom候选对(将触发GICP): {_cand}  "
              f"(阈值{_thr}m, skip_near={self.skip_near_num}, "
              f"层高门控剔除{_cand_dz}对, z_gate={_zgate}m)")
        if _xf_on:
            print(f"  [Diag] 跨楼层景观候选对(远点GICP): {_cand_xf} "
                  f"(dz∈({_zgate}, {_xf_max_dz}]m)")
        print(f"  [Diag] 线程: 外层{self.num_threads} x GICP内层{_gicp_threads} = "
              f"{self.num_threads * _gicp_threads} vs CPU核心{_cores}"
              f"{'  <== 超订!' if self.num_threads * _gicp_threads > _cores else ''}")
        loop_constraints = []  # [(prev_idx, curr_idx, relative_pose_4x4, score), ...]
        step3_no_btc_frames = 0
        step3_search_count = 0
        step3_loop_count = 0
        odom_direct_verify_count = 0
        cross_floor_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有帧到线程池，提交时立即发布 odom 进度（线性显示）
            futures = {}
            for i in range(len(self.keyframe_clouds)):
                # 提交前发布 odom，RViz 显示线性进度
                self._ros_publish_current_odom(i, len(self.keyframe_clouds))
                futures[executor.submit(self._process_frame_step3, i)] = i

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    frame_result = future.result()
                except Exception as e:
                    print(f"  [ERROR] 帧 {i} 处理异常: {e}")
                    continue

                if frame_result is None:
                    continue

                # ==== 在主线程中安全地更新共享状态 ====
                if frame_result.get('no_btc'):
                    step3_no_btc_frames += 1

                if frame_result.get('searched'):
                    step3_search_count += 1

                if frame_result.get('loop_found'):
                    step3_loop_count += 1

                if frame_result.get('odom_direct'):
                    odom_direct_verify_count += 1

                if frame_result.get('cross_floor'):
                    cross_floor_count += 1

                constraint = frame_result.get('constraint')
                if constraint is not None:
                    loop_constraints.append(constraint)

                loop_pair = frame_result.get('loop_pair')
                if loop_pair is not None:
                    self.loop_pairs.append(loop_pair)

        print(f"\n  ----- Step 3 汇总 -----")
        print(f"  搜索帧数: {step3_search_count}")
        print(f"  零BTC帧数: {step3_no_btc_frames}")
        print(f"  Odom直接验证: {odom_direct_verify_count} (odom距离<3m直接GICP)")
        print(f"  BTC检测回环: {step3_loop_count - odom_direct_verify_count}")
        print(f"  检测到回环: {step3_loop_count} (通过验证: {len(loop_constraints)})")
        print(f"  跨楼层景观约束: {cross_floor_count} (非同一层回环，仅修正层间相对位姿)")
        print(f"  总计位姿图约束: {len(loop_constraints)}")
        if loop_constraints:
            print(f"  ---- 通过验证的约束列表 ----")
            for i, constraint in enumerate(loop_constraints):
                # 统一格式: (prev_idx, curr_idx, rel_pose, score, loop_type)
                prev_idx, curr_idx, rel_pose, score, loop_type = constraint
                t = rel_pose[:3, 3]
                if loop_type == 'odom_direct':
                    print(f"  [{i+1}] 帧 {prev_idx} <-> {curr_idx} | odom_direct | "
                          f"平移: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
                elif loop_type == 'cross_floor':
                    print(f"  [{i+1}] 帧 {prev_idx} <-> {curr_idx} | 跨楼层景观 | "
                          f"fitness: {score:.4f} | 平移: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
                else:
                    print(f"  [{i+1}] 帧 {prev_idx} <-> {curr_idx} | BTC | "
                          f"Score: {score:.4f} | 平移: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
        # ===== 回环一致性校验 (PCM): 仅对同层回环生效，跨楼层景观约束不参与 =====
        if getattr(self, 'enable_pcm', False):
            _same = [c for c in loop_constraints if c[4] != 'cross_floor']
            _xf = [c for c in loop_constraints if c[4] == 'cross_floor']
            if len(_same) > 2:
                _same = self._filter_loops_pcm(_same)
            loop_constraints = _same + _xf
        _lap("步骤3 回环检测")

        # ===== 步骤 4: ISAM2 位姿图优化 (对应 C++ ISAM2) =====
        print("\n===== 步骤 4: ISAM2 位姿图优化 =====")
        optimized_poses = self._optimize_pose_graph(loop_constraints)

        # 发布优化后的关键帧轨迹（PGO 输出），与 odom_keyframe_path 对照比较 PGO 效果
        self._ros_publish_path(optimized_poses, self._ros_pub_optimized_path)
        _lap("步骤4 ISAM2优化")

        # ===== 步骤 5: 保存结果 =====
        print("\n===== 步骤 5: 保存结果 =====")
        try:
            self._save_results(optimized_poses)
        finally:
            _lap("步骤5 保存结果")
            _timing_report()

        return optimized_poses

    def _process_frame_step2(self, i):
        """
        处理单个关键帧的BTC描述子生成（Step 2 原子操作），供多线程调用。
        用 self._btc_lock 保护 C++ btc_manager 的数据库操作。

        返回值 dict:
          - num_btcs: int
          - n_pts: int
          - success: bool
          - error: str or None
        """
        result = {'num_btcs': 0, 'n_pts': 0, 'success': False, 'error': None}

        cloud = self.keyframe_clouds[i]
        if len(cloud) < 10:
            result['error'] = f'点数不足: {len(cloud)}'
            return result

        result['n_pts'] = len(cloud)

        try:
            if cloud.shape[1] == 3:
                cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
            else:
                cloud_with_intensity = cloud

            pose = self.keyframe_poses[i]
            frame_position = pose[:3, 3]

            # 只调用一次 GenerateBtcDescs，拿到 BTC 数据后立即释放点云的 numpy 引用
            with self._btc_lock:
                btc_result = self.btc_manager.GenerateBtcDescs(cloud_with_intensity, i)
                num_btcs = btc_result['num_btcs']
                btcs_data = btc_result.get('btcs_data', [])

                # 将生成的 BTC 描述子添加到数据库（用 AddBtcDescsFromCache 跳过重复生成）
                if btcs_data:
                    self.btc_manager.AddBtcDescsFromCache(btcs_data, np.array(frame_position, dtype=np.float64))

            # 尽早释放 cloud_with_intensity 和 btc_result（仅保留必要的 btcs_data）
            del cloud_with_intensity
            del btc_result

            result['num_btcs'] = num_btcs
            result['btcs_data'] = btcs_data
            result['frame_position'] = frame_position.tolist()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def _process_frame_step3(self, i):
        """
        处理单个关键帧的回环检测（Step 3 原子操作），供多线程调用。

        返回值 dict (或 None):
          - constraint: (prev_idx, curr_idx, rel_pose, score, loop_type) or None
          - loop_pair: tuple for self.loop_pairs or None
          - no_btc: bool (BTC=0跳过)
          - searched: bool (执行了搜索)
          - loop_found: bool (找到回环)
          - odom_direct: bool (通过Odom直接验证)

        注意: 此方法不修改 self.loop_pairs / loop_constraints, 但会立即发布ROS话题。
        _save_fused_cloud 是线程安全的文件写入, 可在子线程中调用。
        Python print 是线程安全的(因GIL)，日志可直接输出。
        """
        result = {
            'constraint': None,
            'loop_pair': None,
            'no_btc': False,
            'searched': False,
            'loop_found': False,
            'odom_direct': False,
        }

        cloud = self.keyframe_clouds[i]
        total_kf = len(self.keyframe_clouds)
        if len(cloud) < 10:
            return result

        result['searched'] = True

        pose = self.keyframe_poses[i]
        current_position = pose[:3, 3]

        # ===== Odom距离直接验证（查表优化）=====
        odom_direct_candidates = []
        cross_floor_candidates = []   # 跨楼层候选: (j, odom_dist, dz), 用窗外景观约束
        skip_near_num = self.skip_near_num
        z_i = current_position[2]
        z_gate = getattr(self, 'loop_z_gate', 0.0)
        _xf_on = getattr(self, 'cross_floor_enabled', False)
        _xf_max_dz = getattr(self, 'cross_floor_max_dz', 4.0)
        # 直接从距离矩阵查表获取所有历史帧的距离
        distances = self.keyframe_distance_matrix[i, :i]  # 0 到 i-1 的距离
        for j, odom_distance in enumerate(distances):
            frame_diff = i - j
            if frame_diff <= skip_near_num:
                continue
            if odom_distance < self.odom_direct_threshold:
                # 层高门控: |dz| 超阈值 => 跨楼层(外观相似的楼梯/走廊)，不作为同地点回环
                _dz = abs(z_i - self.keyframe_poses[j][2, 3])
                if z_gate > 0 and _dz > z_gate:
                    # 可能是跨楼层真重叠(窗外共同景观): 记录为跨楼层候选，走独立验证通道
                    if _xf_on and _dz <= _xf_max_dz:
                        cross_floor_candidates.append((j, odom_distance, _dz))
                    continue
                odom_direct_candidates.append((j, odom_distance))

        if odom_direct_candidates and self.use_gicp and HAS_GICP_OMP:
            for (candidate_id, odom_dist) in odom_direct_candidates:
                print(f"  [Odom Direct] 帧 {i} <-> {candidate_id} "
                      f"odom距离 {odom_dist:.2f}m < {self.odom_direct_threshold}m, 直接GICP验证")
                curr_pose = self.keyframe_poses[i]
                prev_pose = self.keyframe_poses[candidate_id]
                relative_pose_matrix = np.linalg.inv(prev_pose) @ curr_pose

                _t0 = time.perf_counter()
                gicp_result = gicp_align(
                    self.keyframe_clouds_ds[i],
                    self.keyframe_clouds_ds[candidate_id],
                    initial_guess=relative_pose_matrix,
                    config=self.gicp_config,
                    compute_overlap=getattr(self, 'save_fused_scans', False)
                )
                _dt_gicp = time.perf_counter() - _t0
                _n_src = len(self.keyframe_clouds_ds[i])
                _n_tgt = len(self.keyframe_clouds_ds[candidate_id])
                _it = int(getattr(gicp_result, 'num_iterations', 0)) if gicp_result else 0
                _ok = bool(gicp_result and gicp_result.has_converged and
                           gicp_result.fitness_score < self.gicp_config.fitness_score_threshold)
                GICP_STATS.add(_dt_gicp, 'odom_direct', _n_src, _n_tgt, _it, _ok)
                _c, _tot = GICP_STATS.snapshot()
                print(f"  [GICP-Timing] {candidate_id}<->{i}: {_dt_gicp:.2f}s | iter={_it} "
                      f"{'OK' if _ok else 'FAIL'} | 累计 #{_c} 次 {_tot:.1f}s | "
                      f"src={_n_src}, tgt={_n_tgt}")

                if gicp_result and gicp_result.has_converged:
                    self._save_fused_cloud(i, candidate_id, gicp_result.transformation,
                                           relative_pose_matrix, gicp_result.fitness_score,
                                           gicp_result.overlap_ratio, 'odom_direct')

                if (gicp_result and gicp_result.has_converged and
                        gicp_result.fitness_score < self.gicp_config.fitness_score_threshold):
                    # 跳过 Degeneracy 检查（加速优化）
                    print(f"  [Odom Direct SUCCESS] 回环验证成功! "
                          f"{candidate_id} <-> {i}, GICP fitness: {gicp_result.fitness_score:.4f}")
                    result['odom_direct'] = True
                    result['loop_found'] = True
                    T_delta = gicp_result.transformation
                    t_vec = T_delta[:3, 3]
                    roll, pitch, yaw = _extract_rpy(T_delta)
                    result['constraint'] = (
                        candidate_id, i, gicp_result.transformation, 0.0, 'odom_direct'
                    )
                    result['loop_pair'] = (
                        candidate_id, i,
                        f"{self.keyframe_indices[candidate_id]:06d}.pcd",
                        f"{self.keyframe_indices[i]:06d}.pcd",
                        0.0, gicp_result.fitness_score,
                        t_vec[0], t_vec[1], t_vec[2],
                        roll, pitch, yaw
                    )
                    # 找到回环立即发布到 RViz
                    self._ros_publish_loop_markers(
                        candidate_id, i, gicp_result.fitness_score,
                        self.keyframe_poses[candidate_id],
                        self.keyframe_poses[i])
                    return result

        # ===== 跨楼层景观约束验证 =====
        # 用窗外共同景观(远点)对齐不同楼层的帧，修正层间相对位姿；
        # 不作为同一层回环，也绝不把两帧拉到同一高度。
        if cross_floor_candidates and self.use_gicp and HAS_GICP_OMP:
            _far_d = self.cross_floor_far_dist
            _min_pts = self.cross_floor_min_pts
            _far_src = _far_range_points(self.keyframe_clouds_ds[i], _far_d)
            if len(_far_src) < _min_pts:
                print(f"  [XFloor] 帧 {i} 远点不足 ({len(_far_src)} < {_min_pts}, >{_far_d}m), "
                      f"跳过 {len(cross_floor_candidates)} 个跨楼层候选")
            else:
                _best = None
                for (candidate_id, odom_dist, _dz) in cross_floor_candidates:
                    _far_tgt = _far_range_points(self.keyframe_clouds_ds[candidate_id], _far_d)
                    if len(_far_tgt) < _min_pts:
                        continue
                    print(f"  [XFloor] 帧 {i} <-> {candidate_id} | dz={_dz:.2f}m | "
                          f"远点 {len(_far_src)}/{len(_far_tgt)} (>={_far_d}m), 仅用远点GICP")
                    relative_pose_matrix = (np.linalg.inv(self.keyframe_poses[candidate_id])
                                            @ self.keyframe_poses[i])
                    _t0 = time.perf_counter()
                    gicp_result = gicp_align(
                        _far_src, _far_tgt,
                        initial_guess=relative_pose_matrix,
                        config=self.gicp_config
                    )
                    _dt_gicp = time.perf_counter() - _t0
                    _it = int(getattr(gicp_result, 'num_iterations', 0)) if gicp_result else 0
                    _fit = getattr(gicp_result, 'fitness_score', 1e9) if gicp_result else 1e9
                    _ok = bool(gicp_result and gicp_result.has_converged and
                               _fit < self.cross_floor_fitness_thres)
                    GICP_STATS.add(_dt_gicp, 'cross_floor', len(_far_src), len(_far_tgt), _it, _ok)
                    print(f"  [GICP-Timing] XFloor {candidate_id}<->{i}: {_dt_gicp:.2f}s | iter={_it} "
                          f"{'OK' if _ok else 'FAIL'} | fitness={_fit:.4f}")
                    if _ok and (_best is None or _fit < _best[3]):
                        _best = (candidate_id, _dz, gicp_result.transformation, _fit)

                if _best is not None:
                    candidate_id, _dz, T_delta, _fit = _best
                    print(f"  [XFloor SUCCESS] 跨楼层景观约束建立: {candidate_id} <-> {i} | "
                          f"dz={_dz:.2f}m | fitness={_fit:.4f} (非同一层回环)")
                    result['cross_floor'] = True
                    result['constraint'] = (
                        candidate_id, i, T_delta, _fit, 'cross_floor'
                    )

        # odom_only 模式：只使用 Odom 直接验证，跳过 BTC/SC 搜索
        if self.use_method == 'odom_only':
            return result

        # ===== BTC匹配流程 =====
        # 从缓存加载 BTC 描述子（避免重复生成，消除 _btc_lock 瓶颈）
        cache_file = self._btc_cache_frame_path(i)
        if os.path.exists(cache_file):
            import pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                btcs_data = cache_data['btcs_data']
                frame_position_np = np.array(cache_data['frame_position'], dtype=np.float64)
                num_btcs = len(btcs_data)

            with self._btc_lock:
                self.btc_manager.AddBtcDescsFromCache(btcs_data, frame_position_np)
        else:
            # 缓存不存在时回退到原方式（理论上不应该发生）
            print(f"  [WARN] 帧 {i} BTC 缓存不存在，回退到重新生成")
            if cloud.shape[1] == 3:
                cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
            else:
                cloud_with_intensity = cloud

            with self._btc_lock:
                btc_result = self.btc_manager.GenerateBtcDescs(cloud_with_intensity, i)
                num_btcs = btc_result['num_btcs']

        if num_btcs == 0:
            result['no_btc'] = True
            print(f"  [Step3] 帧 {i}/{total_kf} | 点云 {len(cloud)} pts | BTC=0, 跳过")
            return result

        # 每50帧或匹配到时打印一次搜索日志
        if i % 50 == 0 or i == total_kf - 1:
            print(f"  [Step3] 帧 {i}/{total_kf} | 点云 {len(cloud)} pts | BTC={num_btcs}, 搜索中...")

        # C++ SearchLoop（仍需 _btc_lock，但时间占比大幅降低）
        # SearchLoop 需要当前帧的点云来计算候选描述子，所以这里仍需点云
        if cloud.shape[1] == 3:
            cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
        else:
            cloud_with_intensity = cloud

        with self._btc_lock:
            search_result = self.btc_manager.SearchLoop(cloud_with_intensity, i, current_position)
        match_frame_id = search_result['match_frame_id']
        loop_score = search_result['match_score']

        if match_frame_id == -1:
            candidates = search_result.get('candidate_frame_ids', [])
            if len(candidates) > 0:
                print(f"  [BTC Diag] frame={i}, {len(candidates)} candidates, none passed verification")
            return result

        t = np.array(search_result['translation'])
        R = np.array(search_result['rotation'])
        relative_pose_matrix = np.eye(4)
        relative_pose_matrix[:3, :3] = R
        relative_pose_matrix[:3, 3] = t

        # GICP 精化
        prev_idx = match_frame_id
        curr_idx = i
        # 层高门控: 跨楼层(外观相似)不作为同地点回环
        if self.loop_z_gate > 0:
            _dz = abs(self.keyframe_poses[curr_idx][2, 3] - self.keyframe_poses[prev_idx][2, 3])
            if _dz > self.loop_z_gate:
                print(f"  [BTC Loop] 跨楼层跳过: {prev_idx} <-> {curr_idx} "
                      f"(dz={_dz:.2f}m > {self.loop_z_gate}m)")
                return result
        print(f"  [BTC Loop] 检测到回环! {prev_idx} <-> {curr_idx}, score: {loop_score:.4f}")
        gicp_success = False
        gicp_fitness = 0.0

        if self.use_gicp and HAS_GICP_OMP:
            curr_pose = self.keyframe_poses[curr_idx]
            prev_pose = self.keyframe_poses[prev_idx]
            odom_relative_pose = np.linalg.inv(prev_pose) @ curr_pose

            init_translation_btc = np.linalg.norm(relative_pose_matrix[:3, 3])
            init_translation_odom = np.linalg.norm(odom_relative_pose[:3, 3])

            if init_translation_odom < init_translation_btc:
                initial_guess = odom_relative_pose
                print(f"  [GICP] 使用odom位姿作为初值 "
                      f"(odom={init_translation_odom:.2f}m < BTC={init_translation_btc:.2f}m)")
            else:
                initial_guess = relative_pose_matrix
                print(f"  [GICP] 使用BTC位姿作为初值 "
                      f"(BTC={init_translation_btc:.2f}m <= odom={init_translation_odom:.2f}m)")

            init_translation = np.linalg.norm(initial_guess[:3, 3])
            if init_translation > self.gicp_config.max_init_translation:
                print(f"  [GICP] 初始平移 {init_translation:.1f}m > "
                      f"{self.gicp_config.max_init_translation}m，跳过")
                return result

            _t0 = time.perf_counter()
            gicp_result = gicp_align(
                self.keyframe_clouds_ds[curr_idx],
                self.keyframe_clouds_ds[prev_idx],
                initial_guess=initial_guess,
                config=self.gicp_config,
                compute_overlap=getattr(self, 'save_fused_scans', False)
            )
            _dt_gicp = time.perf_counter() - _t0
            _n_src = len(self.keyframe_clouds_ds[curr_idx])
            _n_tgt = len(self.keyframe_clouds_ds[prev_idx])
            _it = int(getattr(gicp_result, 'num_iterations', 0)) if gicp_result else 0
            _ok = bool(gicp_result and gicp_result.has_converged and
                       gicp_result.fitness_score < self.gicp_config.fitness_score_threshold)
            GICP_STATS.add(_dt_gicp, 'btc_loop', _n_src, _n_tgt, _it, _ok)
            _c, _tot = GICP_STATS.snapshot()
            print(f"  [GICP-Timing] {prev_idx}<->{curr_idx}: {_dt_gicp:.2f}s | iter={_it} "
                  f"{'OK' if _ok else 'FAIL'} | 累计 #{_c} 次 {_tot:.1f}s | "
                  f"src={_n_src}, tgt={_n_tgt}")

            if (gicp_result and gicp_result.has_converged and
                    gicp_result.fitness_score < self.gicp_config.fitness_score_threshold):
                # 跳过 Degeneracy 检查（加速优化）
                relative_pose_matrix = gicp_result.transformation
                gicp_success = True
                gicp_fitness = gicp_result.fitness_score
                print(f"  [GICP] 精化成功! Fitness: {gicp_result.fitness_score:.4f}")
                self._save_fused_cloud(curr_idx, prev_idx, gicp_result.transformation,
                                       initial_guess, gicp_result.fitness_score,
                                       gicp_result.overlap_ratio, 'btc_loop')
            else:
                fitness = gicp_result.fitness_score if gicp_result else float('inf')
                print(f"  [GICP] 精化失败或分数过高 "
                      f"({fitness:.4f} > {self.gicp_config.fitness_score_threshold})，拒绝回环!")
                return result

            if not gicp_success:
                return result
        else:
            # 无GICP时直接使用BTC位姿
            gicp_success = True

        # 回环验证 (对应 C++ validateLoopClosure)
        if not validate_loop_closure(
                self.keyframe_poses[prev_idx],
                self.keyframe_poses[curr_idx],
                relative_pose_matrix,
                max_loop_distance=self.max_loop_distance,
                max_yaw_diff=self.max_yaw_diff):
            return result

        if self.use_gicp and HAS_GICP_OMP and not gicp_success:
            print(f"  [WARN] GICP失败，拒绝此回环（BTC原始位姿可能错误）")
            return result

        result['loop_found'] = True
        merged_prev = self.keyframe_indices[prev_idx]
        merged_curr = self.keyframe_indices[curr_idx]

        if self.merge_n > 1 and self.merge_indices:
            orig_prev_range = self.merge_indices[merged_prev]
            orig_curr_range = self.merge_indices[merged_curr]
            scan_prev_name = f"{orig_prev_range[0]:06d}-{orig_prev_range[-1]:06d}.pcd"
            scan_curr_name = f"{orig_curr_range[0]:06d}-{orig_curr_range[-1]:06d}.pcd"
        else:
            scan_prev_name = f"{merged_prev:06d}.pcd"
            scan_curr_name = f"{merged_curr:06d}.pcd"

        T_delta = np.linalg.inv(self.keyframe_poses[prev_idx]) @ self.keyframe_poses[curr_idx]
        t_vec = T_delta[:3, 3]
        roll, pitch, yaw = _extract_rpy(T_delta)

        print(f"  ===== 回环验证通过 =====")
        print(f"  KeyFrame: {prev_idx} <-> {curr_idx} | "
              f"Scan {scan_prev_name} <-> {scan_curr_name}")
        print(f"  BTC score: {loop_score:.4f}, GICP fitness: {gicp_fitness:.4f}")
        print(f"  相对平移: [{t_vec[0]:.3f}, {t_vec[1]:.3f}, {t_vec[2]:.3f}] m")
        print(f"  相对旋转 (RPY): [{roll:.3f}, {pitch:.3f}, {yaw:.3f}] rad")
        print(f"  ========================")

        result['constraint'] = (prev_idx, curr_idx, relative_pose_matrix, loop_score, 'btc')
        result['loop_pair'] = (
            prev_idx, curr_idx,
            scan_prev_name, scan_curr_name,
            loop_score, gicp_fitness,
            t_vec[0], t_vec[1], t_vec[2],
            roll, pitch, yaw
        )
        # 找到回环立即发布到 RViz
        self._ros_publish_loop_markers(
            prev_idx, curr_idx, loop_score,
            self.keyframe_poses[prev_idx],
            self.keyframe_poses[curr_idx])

        return result

    def _select_keyframes(self):
        """关键帧选择，与 C++ process_pg 中的逻辑一致"""
        # 尝试从缓存加载关键帧索引
        import json
        kf_cache_file = os.path.join(self._btc_cache_dir(), "keyframes.json")
        kf_cache = None
        if os.path.exists(kf_cache_file):
            try:
                with open(kf_cache_file, 'r') as f:
                    kf_cache = json.load(f)
            except Exception:
                pass

        if (kf_cache and
                kf_cache.get('total_frames') == len(self.all_poses) and
                kf_cache.get('meter_gap') == self.keyframe_meter_gap and
                kf_cache.get('deg_gap') == self.keyframe_deg_gap and
                isinstance(kf_cache.get('indices'), list)):
            print(f"[Cache] 关键帧缓存命中: {len(kf_cache['indices'])} 帧 ({len(self.all_poses)} merged 帧)")
            kf_indices = kf_cache['indices']
        else:
            kf_indices = self._compute_keyframe_indices()

            # 写入缓存
            cache_dir = self._btc_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            kf_data = {
                'total_frames': len(self.all_poses),
                'meter_gap': self.keyframe_meter_gap,
                'deg_gap': self.keyframe_deg_gap,
                'indices': kf_indices,
            }
            with open(kf_cache_file, 'w') as f:
                json.dump(kf_data, f)
            print(f"[Cache] 关键帧缓存已保存: {kf_cache_file} ({len(kf_indices)} 帧)")

        # 根据索引构建关键帧数据结构（从 merged all_poses/all_scans）
        self._build_keyframes_from_indices(kf_indices)

    def _compute_keyframe_indices(self):
        """计算关键帧索引列表"""
        indices = []
        translation_accumulated = float('inf')
        rotation_accumulated = float('inf')
        prev_pose6d = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(len(self.all_poses)):
            T = self.all_poses[i]
            pose6d = self._matrix_to_pose6d(T)

            T_prev = _pose6d_to_matrix(prev_pose6d)
            T_curr = _pose6d_to_matrix(pose6d)
            T_delta = np.linalg.inv(T_prev) @ T_curr

            dx = abs(T_delta[0, 3])
            dy = abs(T_delta[1, 3])
            dz = abs(T_delta[2, 3])
            delta_translation = np.sqrt(dx**2 + dy**2 + dz**2)
            roll_d, pitch_d, yaw_d = _extract_rpy(T_delta)
            delta_rotation = abs(roll_d) + abs(pitch_d) + abs(yaw_d)

            translation_accumulated += delta_translation
            rotation_accumulated += delta_rotation

            keyframe_rad_gap = np.deg2rad(self.keyframe_deg_gap)
            is_key_frame = (translation_accumulated > self.keyframe_meter_gap or
                            rotation_accumulated > keyframe_rad_gap)

            if is_key_frame or i == 0:
                translation_accumulated = 0.0
                rotation_accumulated = 0.0
                indices.append(i)

            prev_pose6d = pose6d

        return indices

    def _build_keyframes_from_indices(self, kf_indices):
        """根据索引从 merged all_poses/all_scans 构建关键帧数据结构"""
        self.keyframe_clouds = []
        self.keyframe_clouds_ds = []
        self.keyframe_poses = []
        self.keyframe_poses6d = []
        self.keyframe_indices = []

        _t_ds = 0.0
        _pts_in = 0
        _pts_out = 0
        for i in kf_indices:
            T = self.all_poses[i]
            pose6d = self._matrix_to_pose6d(T)
            cloud = self.all_scans[i]

            if len(cloud) > 0:
                _t0 = time.perf_counter()
                cloud_ds = down_sampling_voxel(cloud, self.scan_ds_size)
                _t_ds += time.perf_counter() - _t0
                _pts_in += len(cloud)
                _pts_out += len(cloud_ds)
            else:
                cloud_ds = cloud

            self.keyframe_clouds.append(cloud)       # BTC 用原始点云
            self.keyframe_clouds_ds.append(cloud_ds)  # GICP 用下采样点云
            self.keyframe_poses.append(T.copy())
            self.keyframe_poses6d.append(pose6d)
            self.keyframe_indices.append(i)

        print(f"  [KFrame-Timing] 体素下采样 {len(kf_indices)} 帧: {_t_ds:.2f}s | "
              f"输入点 {_pts_in:,} → 输出点 {_pts_out:,} (voxel={self.scan_ds_size})")

    def _filter_loops_pcm(self, loop_constraints):
        """
        回环一致性校验 (Pairwise Consistency Maximization 简化版)。

        以里程计链为参考，检查任意两个回环组成的闭合环残差：
            T1 @ O(b→c) @ T2 @ O(d→a) ≈ I
        贪心保留“最大的一致子集”，剔除与群体不一致的孤立/错误回环
        （典型：多楼层外观相似的楼梯被误当作同一地点）。

        loop_constraints: [(prev_idx, curr_idx, rel_pose_4x4, score, loop_type), ...]
        返回: 过滤后的 loop_constraints
        """
        n = len(loop_constraints)
        if n <= 2:
            return loop_constraints

        pos_thres = getattr(self, 'pcm_pos_thres', 1.5)
        rot_thres = getattr(self, 'pcm_rot_thres', 0.2)
        P = self.keyframe_poses

        def _consistent(l1, l2):
            a, b, T1 = l1[0], l1[1], l1[2]
            c, d, T2 = l2[0], l2[1], l2[2]
            # gtsam 约定: delta = inv(pose_i) @ pose_j
            O_bc = np.linalg.inv(P[b]) @ P[c]
            O_da = np.linalg.inv(P[d]) @ P[a]
            C = T1 @ O_bc @ T2 @ O_da
            dt = float(np.linalg.norm(C[:3, 3]))
            return dt < pos_thres and _rot_angle_rad(C[:3, :3]) < rot_thres

        # 按 score 升序（越小越好）贪心选兼容子集
        order = sorted(range(n), key=lambda k: loop_constraints[k][3])
        kept = []
        for k in order:
            if all(_consistent(loop_constraints[k], loop_constraints[m]) for m in kept):
                kept.append(k)
        kept.sort()
        dropped = n - len(kept)
        if dropped:
            print(f"  [PCM] 一致性校验: 保留 {len(kept)}/{n} 个回环 "
                  f"(剔除 {dropped} 个不一致回环, pos<{pos_thres}m, rot<{rot_thres}rad)")
        else:
            print(f"  [PCM] 一致性校验: 全部 {n} 个回环通过")
        return [loop_constraints[k] for k in kept]

    def _optimize_pose_graph(self, loop_constraints):
        """
        ISAM2 位姿图优化，与 C++ 版本一致:
          - prior noise: 1e-12
          - odom noise: [1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]
          - loop noise: 鲁棒核(loop_robust_kernel, 默认 Huber) + Diagonal(1.0/0.25)
        """
        if not HAS_GTSAM:
            print("[SKIP] GTSAM 不可用，跳过位姿图优化")
            return self.keyframe_poses

        if len(self.keyframe_poses) == 0:
            return self.keyframe_poses

        prior_noise, odom_noise, robust_loop_noise, robust_gps_noise = init_noises(
            getattr(self, 'loop_robust_kernel', 'huber'),
            getattr(self, 'loop_robust_k', 0.0))

        # ISAM2 参数 (对应 C++ ISAM2Params)
        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.01)
        isam_params.relinearizeSkip = 1
        isam = gtsam.ISAM2(isam_params)

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # 先验节点
        pose0 = matrix_to_gtsam_pose3(self.keyframe_poses[0])
        graph.add(gtsam.PriorFactorPose3(0, pose0, prior_noise))
        initial.insert(0, pose0)

        # 里程计因子
        for i in range(1, len(self.keyframe_poses)):
            pose_i = matrix_to_gtsam_pose3(self.keyframe_poses[i])
            initial.insert(i, pose_i)

            T_delta = np.linalg.inv(self.keyframe_poses[i - 1]) @ self.keyframe_poses[i]
            delta = matrix_to_gtsam_pose3(T_delta)
            graph.add(gtsam.BetweenFactorPose3(i - 1, i, delta, odom_noise))

        # 回环因子
        for constraint in loop_constraints:
            # 统一格式: (prev_idx, curr_idx, rel_pose, score, loop_type)
            prev_idx, curr_idx, rel_pose_mat, score, loop_type = constraint

            rel_gtsam = matrix_to_gtsam_pose3(rel_pose_mat)
            graph.add(gtsam.BetweenFactorPose3(prev_idx, curr_idx, rel_gtsam, robust_loop_noise))
            if loop_type == 'cross_floor':
                print(f"  添加跨楼层景观约束: {prev_idx} <-> {curr_idx}")
            else:
                print(f"  添加回环因子: {prev_idx} <-> {curr_idx}")

        # 离线优化使用 Levenberg-Marquardt 批量优化器
        # ISAM2 是增量式优化器，适合在线 SLAM；LM 适合小规模离线批量优化
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        # 提取优化后的位姿
        optimized_poses = []
        for i in range(len(self.keyframe_poses)):
            if result.exists(i):
                pose3 = result.atPose3(i)
                optimized_poses.append(gtsam_pose3_to_matrix(pose3))
            else:
                optimized_poses.append(self.keyframe_poses[i].copy())

        print(f"  ISAM2 优化完成, {len(optimized_poses)} 个位姿")
        return optimized_poses

    def _save_results(self, optimized_poses):
        """保存结果 — 将关键帧优化结果传播到所有原始帧"""
        data_dir = self.data_dir

        # 将关键帧的优化校正传播到所有（合并后的）帧
        # 正确做法：odom相对位姿是固定的，把关键帧的校正量通过odom约束传播
        #   odom_delta = inv(T_orig_kf) @ T_orig_i     (kf→i 的刚性 odom 变换)
        #   T_opt_i = T_opt_kf @ odom_delta            (在优化后的kf上叠加刚性变换)
        # 无需插值，因为 odom_delta 本身已在同一 odom 坐标系下定义，几何正确。
        full_optimized_merged = []
        for i, T_orig in enumerate(self.all_poses):
            # 找到当前帧前后的关键帧索引
            kf_idx_before = None
            kf_idx_after = None
            for ki, oi in enumerate(self.keyframe_indices):
                if oi <= i:
                    kf_idx_before = ki
                if oi >= i and kf_idx_after is None:
                    kf_idx_after = ki

            if kf_idx_before is None:
                kf_idx_before = 0
            if kf_idx_after is None:
                kf_idx_after = len(self.keyframe_indices) - 1

            # odom 刚性约束：从关键帧到帧 i 的相对变换
            T_orig_kf = self.keyframe_poses[kf_idx_before]
            odom_delta = np.linalg.inv(T_orig_kf) @ T_orig
            # 在优化后的关键帧位姿上叠加 odom 约束
            T_opt = optimized_poses[kf_idx_before] @ odom_delta

            full_optimized_merged.append(T_opt)

        # 如果使用了多帧合并，将校正回推到原始帧
        # 正确做法：合并帧的参考坐标系是组内最后一帧，odom相对变换固定
        #   T_opt_orig_i = T_merged_opt @ inv(T_merged_orig) @ T_orig_i
        if self.merge_n > 1 and self.original_poses is not None:
            print(f"  将优化结果回推到 {len(self.original_poses)} 个原始帧...")
            full_optimized = []
            for g, T_merged_opt in enumerate(full_optimized_merged):
                T_merged_orig = self.all_poses[g]  # 合并帧的原始 odom 位姿 (组内最后一帧)

                for orig_idx in self.merge_indices[g]:
                    T_orig_i = self.original_poses[orig_idx]
                    odom_delta = np.linalg.inv(T_merged_orig) @ T_orig_i
                    T_opt_i = T_merged_opt @ odom_delta
                    full_optimized.append(T_opt_i)
        else:
            full_optimized = full_optimized_merged

        # 保存全部原始帧的优化轨迹
        opt_file = os.path.join(data_dir, "optimized_poses.txt")
        self._save_poses_kitti(opt_file, full_optimized)
        print(f"  优化轨迹已保存: {opt_file} ({len(full_optimized)} 帧)")

        # 回环对 — 将合并帧索引映射回原始帧索引
        pairs_file = os.path.join(data_dir, "loop_pairs.txt")
        with open(pairs_file, 'w') as f:
            f.write("# frame_a frame_b scan_file_a scan_file_b btc_score fitness_score tx ty tz roll pitch yaw\n")
            for pair in self.loop_pairs:
                if len(pair) >= 12:
                    prev_kf, curr_kf, scan_a, scan_b, score, fitness, tx, ty, tz, roll, pitch, yaw = pair
                    # 合并模式下将合并帧索引映射回原始帧索引
                    if self.merge_n > 1 and self.merge_indices and hasattr(self, 'keyframe_indices'):
                        merged_prev = self.keyframe_indices[prev_kf] if prev_kf < len(self.keyframe_indices) else prev_kf
                        merged_curr = self.keyframe_indices[curr_kf] if curr_kf < len(self.keyframe_indices) else curr_kf
                        orig_prev = self.merge_indices[merged_prev][-1] if merged_prev < len(self.merge_indices) else merged_prev
                        orig_curr = self.merge_indices[merged_curr][-1] if merged_curr < len(self.merge_indices) else merged_curr
                        scan_a = f"{orig_prev:06d}.pcd"
                        scan_b = f"{orig_curr:06d}.pcd"
                    f.write(f"{prev_kf} {curr_kf} {scan_a} {scan_b} "
                            f"{score:.6f} {fitness:.6f} "
                            f"{tx:.4f} {ty:.4f} {tz:.4f} "
                            f"{roll:.4f} {pitch:.4f} {yaw:.4f}\n")
                else:
                    # 兼容旧格式
                    a, b, s, fs = pair
                    f.write(f"{a} {b} -1 -1 {s:.6f} {fs:.6f} 0 0 0 0 0 0\n")
        print(f"  回环对已保存: {pairs_file}")

    # ======================== 融合点云输出 ========================

    @staticmethod
    def _rot_to_euler(R):
        """4x4或3x3旋转矩阵 → 欧拉角(roll,pitch,yaw) 度"""
        r = R[:3, :3] if R.shape[0] >= 4 else R
        sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)
        if sy > 1e-6:
            roll = np.arctan2(r[2, 1], r[2, 2])
            pitch = np.arctan2(-r[2, 0], sy)
            yaw = np.arctan2(r[1, 0], r[0, 0])
        else:
            roll = np.arctan2(-r[1, 2], r[1, 1])
            pitch = np.arctan2(-r[2, 0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    def _save_fused_cloud(self, curr_idx, prev_idx, gicp_T, odom_T, gicp_fitness, gicp_overlap, tag='loop'):
        """
        将GICP匹配后的两帧点云融合输出到FusedScans/目录供目视检查。

        用 intensity 区分不同帧：
          - prev帧: intensity=50（暗）
          - curr帧: intensity=200（亮）
        CloudCompare 中可用 Scalar Field → intensity 来着色显示。

        gicp_T: 4x4, GICP精化后的curr→prev变换
        odom_T: 4x4, odom推算的curr→prev变换（初值）
        gicp_fitness: float, GICP内点平均距离 (m)
         gicp_overlap: float, overlap ratio (0~1)
        """
        fused_dir = os.path.join(self.data_dir, "FusedScans")

        if not getattr(self, 'save_fused_scans', False):
            print(f"  [Fused] 已跳过 (save_fused_scans=false): {tag} {prev_idx}↔{curr_idx}")
            return

        curr_cloud = self.keyframe_clouds_ds[curr_idx]
        prev_cloud = self.keyframe_clouds_ds[prev_idx]

        # === GICP结果融合（intensity标记帧来源） ===
        R_gicp = gicp_T[:3, :3]
        t_gicp = gicp_T[:3, 3]
        curr_to_prev = (R_gicp @ curr_cloud.T).T + t_gicp
        fused_pts_gicp = np.vstack([prev_cloud, curr_to_prev])
        fused_int_gicp = np.hstack([
            np.full(len(prev_cloud), 50, dtype=np.float32),   # prev: 暗
            np.full(len(curr_cloud), 200, dtype=np.float32)   # curr: 亮
        ])

        # === Odom初值融合（intensity标记帧来源） ===
        R_odom = odom_T[:3, :3]
        t_odom = odom_T[:3, 3]
        curr_to_prev_odom = (R_odom @ curr_cloud.T).T + t_odom
        fused_pts_odom = np.vstack([prev_cloud, curr_to_prev_odom])
        fused_int_odom = np.hstack([
            np.full(len(prev_cloud), 50, dtype=np.float32),
            np.full(len(curr_cloud), 200, dtype=np.float32)
        ])

        gicp_disp = np.linalg.norm(t_gicp - t_odom)

        gicp_path = os.path.join(fused_dir,
            f"{tag}_fit{gicp_fitness:.2f}m_ovl{gicp_overlap:.0%}_{prev_idx:04d}_to_{curr_idx:04d}_gicp.pcd")
        odom_path = os.path.join(fused_dir,
            f"{tag}_fit{gicp_fitness:.2f}m_ovl{gicp_overlap:.0%}_{prev_idx:04d}_to_{curr_idx:04d}_odom.pcd")

        # FusedScans 仅用于目视检查，写入失败不应影响回环结果
        try:
            os.makedirs(fused_dir, exist_ok=True)
            _save_pcd_numpy_intensity(gicp_path, fused_pts_gicp, fused_int_gicp)
            _save_pcd_numpy_intensity(odom_path, fused_pts_odom, fused_int_odom)
        except Exception as e:
            print(f"  [WARN] FusedScans 写入失败 ({tag} {prev_idx}↔{curr_idx})，"
                  f"已跳过可视化输出，不影响回环结果: {e}")
            return

        print(f"  [Fused] 已保存: {tag} {prev_idx}↔{curr_idx}")
        print(f"          GICP位移={np.linalg.norm(t_gicp):.3f}m, Odom位移={np.linalg.norm(t_odom):.3f}m, "
              f"差异={gicp_disp:.3f}m, mean_dist={gicp_fitness:.2f}m, overlap={gicp_overlap:.0%}")


    # ======================== I/O 工具 ========================

    @staticmethod
    def _load_poses_kitti(filepath):
        """
        加载 KITTI 格式位姿 (每行12列)
        KITTI格式: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        对应矩阵: [R row0 | t[0]]
                  [R row1 | t[1]]
                  [R row2 | t[2]]
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(v) for v in line.split()]
                if len(vals) != 12:
                    continue

                # 正确解析：提取各行数据
                # vals[0:4]: R的第0行 + t[0]
                # vals[4:8]: R的第1行 + t[1]
                # vals[8:12]: R的第2行 + t[2]
                row0 = vals[0:4]  # r11 r12 r13 tx
                row1 = vals[4:8]  # r21 r22 r23 ty
                row2 = vals[8:12] # r31 r32 r33 tz

                # 构建4x4变换矩阵
                T = np.array([
                    [row0[0], row0[1], row0[2], row0[3]],
                    [row1[0], row1[1], row1[2], row1[3]],
                    [row2[0], row2[1], row2[2], row2[3]],
                    [0,      0,      0,      1]
                ])
                poses.append(T)
        return poses

    @staticmethod
    def _save_poses_kitti(filepath, poses):
        """保存 KITTI 格式位姿"""
        with open(filepath, 'w') as f:
            for T in poses:
                R = T[:3, :3]
                t = T[:3, 3]
                f.write(f"{R[0,0]:.6f} {R[0,1]:.6f} {R[0,2]:.6f} {t[0]:.6f} "
                        f"{R[1,0]:.6f} {R[1,1]:.6f} {R[1,2]:.6f} {t[1]:.6f} "
                        f"{R[2,0]:.6f} {R[2,1]:.6f} {R[2,2]:.6f} {t[2]:.6f}\n")

    @staticmethod
    def _load_pcd(filepath):
        """加载 PCD 点云文件，过滤 NaN/Inf"""
        pts = _load_pcd_numpy(filepath)
        # 过滤无效点
        if len(pts) > 0:
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]
        return pts

    @staticmethod
    def _matrix_to_pose6d(T):
        """4x4 -> (x,y,z,roll,pitch,yaw)"""
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        roll, pitch, yaw = _extract_rpy(T)
        return (x, y, z, roll, pitch, yaw)


def _load_pcd_numpy(filepath):
    """不依赖 open3d 的 PCD 加载，支持 ASCII 和 Binary 格式"""
    try:
        import struct
        
        # 解析header
        with open(filepath, 'rb') as f:
            header = {}
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                
                if line.startswith('FIELDS'):
                    header['fields'] = line.split()[1:]
                elif line.startswith('SIZE'):
                    header['sizes'] = [int(s) for s in line.split()[1:]]
                elif line.startswith('TYPE'):
                    header['types'] = line.split()[1:]
                elif line.startswith('WIDTH'):
                    header['width'] = int(line.split()[1])
                elif line.startswith('HEIGHT'):
                    header['height'] = int(line.split()[1])
                elif line.startswith('POINTS'):
                    header['points'] = int(line.split()[1])
                elif line.startswith('DATA'):
                    header['data_type'] = line.split()[1].lower()
                    break
            
            # 计算每个点的字节大小
            point_step = sum(header.get('sizes', [4, 4, 4]))  # 默认xyz各4字节
            total_points = header.get('points', header.get('width', 0))
            
            # 读取数据
            if header['data_type'] == 'ascii':
                # ASCII格式：重新打开文件读取文本数据
                points = []
                with open(filepath, 'r') as f_text:
                    in_data = False
                    for line in f_text:
                        if line.strip().startswith('DATA'):
                            in_data = True
                            continue
                        if in_data:
                            vals = line.strip().split()
                            if len(vals) >= 3:
                                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                return np.array(points)
            
            else:  # binary格式
                # 定位到数据起始位置（header结束后）
                header_size = sum(len(l) + 1 for l in header_lines)  # +1 for newline
                f.seek(header_size)
                
                # 读取二进制数据
                binary_data = f.read(total_points * point_step)
                
                # 解析点云（假设FIELDS至少包含x,y,z）
                fields = header.get('fields', ['x', 'y', 'z'])
                sizes = header.get('sizes', [4, 4, 4])
                types = header.get('types', ['F', 'F', 'F'])
                
                # 找出x,y,z字段的位置和类型
                x_idx = fields.index('x') if 'x' in fields else 0
                y_idx = fields.index('y') if 'y' in fields else 1
                z_idx = fields.index('z') if 'z' in fields else 2
                
                # 解析数据
                points = []
                offset = 0
                for i in range(total_points):
                    point_bytes = binary_data[offset:offset+point_step]
                    
                    # 计算每个字段在point内的偏移量
                    field_offsets = []
                    cum_offset = 0
                    for s in sizes:
                        field_offsets.append(cum_offset)
                        cum_offset += s
                    
                    # 提取x,y,z值
                    x_bytes = point_bytes[field_offsets[x_idx]:field_offsets[x_idx]+sizes[x_idx]]
                    y_bytes = point_bytes[field_offsets[y_idx]:field_offsets[y_idx]+sizes[y_idx]]
                    z_bytes = point_bytes[field_offsets[z_idx]:field_offsets[z_idx]+sizes[z_idx]]
                    
                    x = struct.unpack('<f', x_bytes)[0] if types[x_idx] == 'F' else 0.0
                    y = struct.unpack('<f', y_bytes)[0] if types[y_idx] == 'F' else 0.0
                    z = struct.unpack('<f', z_bytes)[0] if types[z_idx] == 'F' else 0.0
                    
                    points.append([x, y, z])
                    offset += point_step
                
                return np.array(points, dtype=np.float32)
                
    except Exception as e:
        print(f"[ERROR] 加载 PCD 失败: {filepath}, {e}")
        import traceback
        traceback.print_exc()
        return np.empty((0, 3))


def _save_pcd_numpy(filepath, pts):
    """保存 Nx3 numpy 点云为 ASCII PCD 文件（不依赖 open3d）"""
    with open(filepath, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(pts)}\n")
        f.write("DATA ascii\n")
        for pt in pts:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")


def _save_pcd_numpy_intensity(filepath, pts, intensity):
    """保存 Nx3 + intensity 为 ASCII PCD（CloudCompare可读取intensity field着色）"""
    with open(filepath, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(pts)}\n")
        f.write("DATA ascii\n")
        for pt, iv in zip(pts, intensity):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {iv:.1f}\n")
