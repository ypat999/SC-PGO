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
import numpy as np
from numpy import linalg as LA
import yaml

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
        self.transformation_epsilon = 1e-8       # 收敛精度（变换增量阈值）
        self.max_correspondence_distance = 2.0   # 最大对应点距离 (m)，回环初始误差大时不宜太大
        self.rotation_epsilon = 0.002
        self.k_correspondences = 20
        self.max_optimizer_iterations = 20
        self.gicp_epsilon = 0.001                # GICP 协方差正则化（防奇异）
        self.max_iterations = 100                 # 每级最大迭代次数
        self.fitness_score_threshold = 0.15        # PCL原生fitness(1m)，优秀匹配<0.01
        self.num_threads = 4
        # 多级配准参数
        self.coarse_ds_size = 0.3                 # 粗配准下采样体素
        self.coarse_max_iter = 50                 # 粗配准最大迭代
        self.coarse_max_dist = 3.0                # 粗配准最大对应距离
        # 新增：GICP初始平移阈值
        self.max_init_translation = 15.0          # 初始平移超过此值时跳过验证


class GICPResult:
    """对应 C++ GICPResult"""
    def __init__(self):
        self.transformation = np.eye(4)
        self.has_converged = False
        self.fitness_score = float('inf')      # mean_inlier_dist (m)，越小越好
        self.overlap_ratio = 0.0               # 0~1，越大越好
        self.num_iterations = 0


def gicp_align(source_pts, target_pts, initial_guess=None, config=None):
    """
    pclomp GICP_OMP 配准 — 与在线 C++ 定位使用完全相同的 pclomp::GeneralizedIterativeClosestPoint。

    - fitness_score 量级与在线一致（0.01~0.05 级别）
    - 协方差计算 + KDTree+BFGS 均 OpenMP 并行
    - 内部自动从 KNN 邻居计算协方差矩阵，无需手动估算法向量

    source_pts: Nx3 numpy array (或 Nx4 with intensity)
    target_pts: Nx3 numpy array (或 Nx4 with intensity)
    initial_guess: 4x4 numpy array
    config: GICPConfig
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
        gicp = gicp_omp_cpp.GicpOmpManager()
        gicp_config = {
            'max_correspondence_distance': config.max_correspondence_distance,
            'transformation_epsilon': config.transformation_epsilon,
            'rotation_epsilon': config.rotation_epsilon,
            'k_correspondences': config.k_correspondences,
            'max_optimizer_iterations': config.max_optimizer_iterations,
            'gicp_epsilon': config.gicp_epsilon,
            'max_iterations': config.max_iterations,
            'num_threads': config.num_threads,
        }
        gicp.setConfig(gicp_config)

        res = gicp.alignTwoStage(
            src_nx4, tgt_nx4, initial_guess.astype(np.float64),
            coarse_ds_size=config.coarse_ds_size,
            coarse_max_iter=config.coarse_max_iter,
            coarse_max_dist=config.coarse_max_dist,
            fine_max_iter=config.max_iterations,
            fine_max_dist=config.max_correspondence_distance,
        )

        result.transformation = np.asarray(res['transformation'])
        result.has_converged = bool(res['has_converged'])
        result.fitness_score = float(res['fitness_score'])          # mean_inlier_dist (m)
        result.overlap_ratio = float(res.get('overlap_ratio', 0.0))

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

def init_noises():
    """
    对应 C++ initNoises
    返回 (prior_noise, odom_noise, robust_loop_noise, robust_gps_noise)

    噪声模型说明（odom 可信场景）:
      - prior_noise: 先验噪声极小，固定首个位姿
      - odom_noise: 里程计噪声极小（sigma平移1cm, 旋转0.5°），坚信odom
      - loop_noise: 回环噪声较大（sigma平移1m, 旋转0.5rad），不信任回环
        配合 Huber 鲁棒核，仅当回环与 odom 高度一致时才起作用
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
    robust_loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Variances(loop_noise_vector)
    )
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
    """
    with open(config_file, 'r') as f:
        lines = f.readlines()
    # 跳过 OpenCV FileStorage 头 (%YAML:1.0)
    cleaned = ''.join(l for l in lines if not l.strip().startswith('%'))
    data = yaml.safe_load(cleaned)

    result = {}
    result['btc_config_path'] = config_file

    # GICP 部分
    gicp_cfg = GICPConfig()
    g = data.get('gicp', {})
    gicp_cfg.fitness_score_threshold = g.get('fitness_score_threshold', 0.3)
    gicp_cfg.max_correspondence_distance = g.get('max_correspondence_distance', 2.0)
    gicp_cfg.max_iterations = g.get('max_iterations', 100)
    gicp_cfg.transformation_epsilon = g.get('transformation_epsilon', 1e-8)
    gicp_cfg.gicp_epsilon = g.get('gicp_epsilon', 0.001)
    gicp_cfg.scan_ds_size = g.get('scan_ds_size', 0.1)
    # 粗配准参数
    gicp_cfg.coarse_ds_size = g.get('coarse_ds_size', 0.3)
    gicp_cfg.coarse_max_iter = g.get('coarse_max_iter', 50)
    gicp_cfg.coarse_max_dist = g.get('coarse_max_dist', 3.0)
    # 新增：GICP初始平移阈值
    gicp_cfg.max_init_translation = g.get('max_init_translation', 15.0)  # 从yaml读取
    result['gicp_config'] = gicp_cfg

    # BTC部分 - 新增读取skip_near_num
    btc_cfg = data.get('btc', {})
    result['skip_near_num'] = btc_cfg.get('skip_near_num', 5)  # 从配置文件读取

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
    result['max_loop_distance'] = lv.get('max_distance', 100.0)
    result['max_yaw_diff'] = lv.get('max_yaw_diff', np.pi * 0.75)
    result['odom_direct_threshold'] = lv.get('odom_direct_threshold', 3.0)

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
                 merge_n=1):
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
        """
        self.data_dir = data_dir or "/home/ywj/save_data/"
        self.debug_btc = debug_btc
        self.merge_n = max(1, int(merge_n))
        self.odom_direct_threshold = odom_direct_threshold
        self.skip_near_num = skip_near_num
        self.use_method = use_method  # 回环检测方法选择

        # BTC C++ 模块
        if HAS_BTC_CPP and use_method == 'btc':
            import btc_cpp
            if btc_config_file and os.path.exists(btc_config_file):
                self.btc_cpp = btc_cpp.BtcDescManager(btc_config_file)
                print(f"[BTC] ✓ 使用C++实现 (加载配置): {btc_config_file}")
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
            if btc_config_file and os.path.exists(btc_config_file):
                self.btc_cpp = btc_cpp.BtcDescManager(btc_config_file)
                print(f"[BTC] 使用C++实现 (加载配置): {btc_config_file}")
            else:
                self.btc_cpp = btc_cpp.BtcDescManager()
                print("[BTC] 使用C++实现 (内置默认配置)")

            # 启用C++ debug日志
            if self.debug_btc and hasattr(self.btc_cpp, 'SetDebugInfo'):
                self.btc_cpp.SetDebugInfo(True)

            # 设置最大回环距离阈值（用于预过滤候选帧）
            if hasattr(self.btc_cpp, 'SetMaxLoopDistance'):
                self.btc_cpp.SetMaxLoopDistance(max_loop_distance)
                print(f"[BTC] 最大回环距离阈值: {max_loop_distance}m")

        self.btc_manager = self.btc_cpp  # 统一接口
        self.btc_config = None  # Python不再需要BTC配置对象

        # 回环验证参数
        self.max_loop_distance = max_loop_distance
        if max_yaw_diff is not None:
            self.max_yaw_diff = max_yaw_diff
        else:
            self.max_yaw_diff = np.pi * 0.75  # 默认 0.75π
            print("[BTC] Python BTC调试日志已开启")

        # GICP 配置
        self.gicp_config = gicp_config or GICPConfig()
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

    def load_data(self, data_dir=None):
        """加载位姿和点云数据"""
        data_dir = data_dir or self.data_dir

        # 加载位姿
        poses_file = os.path.join(data_dir, "odom_poses.txt")
        if not os.path.exists(poses_file):
            print(f"[ERROR] 位姿文件不存在: {poses_file}")
            return False

        self.all_poses = self._load_poses_kitti(poses_file)
        print(f"[Load] 加载了 {len(self.all_poses)} 个位姿")

        # 加载点云
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

        # 备份原始数据
        self.original_poses = [T.copy() for T in self.all_poses]
        self.original_scans = list(self.all_scans)

        merged_poses = []
        merged_scans = []
        self.merge_indices = []

        num_groups = (total + n - 1) // n
        print(f"\n===== 多帧合并 (每{n}帧→1帧) =====")
        print(f"  原始帧数: {total}, 合并组数: {num_groups}")

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

    def run(self):
        """执行完整的离线回环流程"""
        if not hasattr(self, 'all_poses'):
            print("[ERROR] 请先调用 load_data()")
            return

        # ===== 步骤 0: 多帧合并 (可选) =====
        self._merge_frames()

        # ===== 步骤 1: 关键帧选择 (对应 C++ process_pg 中的关键帧判断) =====
        print("\n===== 步骤 1: 关键帧选择 =====")
        self._select_keyframes()
        print(f"  关键帧数: {len(self.keyframe_poses)} / {len(self.all_poses)}")

        if len(self.keyframe_poses) < 20:
            print("[WARN] 关键帧数不足 20，无法进行回环检测")
            return

        # ===== 步骤 2: BTC 描述子生成 + 数据库构建 =====
        print("\n===== 步骤 2: BTC 描述子生成 =====")
        total_btcs = 0
        zero_btc_frames = 0
        for i in range(len(self.keyframe_clouds)):
            cloud = self.keyframe_clouds[i]
            if len(cloud) < 10:
                zero_btc_frames += 1
                if zero_btc_frames <= 5 or i == len(self.keyframe_clouds) - 1:
                    print(f"  [WARN] 关键帧 {i} 点云点数不足: {len(cloud)}，跳过")
                continue

            try:
                # C++ BTC: AddBtcDescs需要Nx4 numpy数组
                # 将Nx3点云转换为Nx4（添加intensity）
                if cloud.shape[1] == 3:
                    cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
                else:
                    cloud_with_intensity = cloud

                # 提取关键帧odom位置（用于预过滤候选帧）
                pose = self.keyframe_poses[i]
                frame_position = pose[:3, 3]  # 提取平移部分 (x, y, z)

                # 添加到数据库（传入帧位置）
                self.btc_manager.AddBtcDescs(cloud_with_intensity, i, frame_position)
                # 获取BTC数量
                result = self.btc_manager.GenerateBtcDescs(cloud_with_intensity, i)
                num_btcs = result['num_btcs']

                total_btcs += num_btcs
                if num_btcs == 0:
                    zero_btc_frames += 1

                if i % 10 == 0 or i == len(self.keyframe_clouds) - 1 or num_btcs == 0:
                    print(f"  关键帧 {i}/{len(self.keyframe_clouds)} | 点云: {len(cloud)} pts | BTC: {num_btcs} | 累计: {total_btcs}")
                    
            except Exception as e:
                print(f"  [ERROR] 关键帧 {i} BTC生成失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Step 2 汇总
        print(f"\n  ----- Step 2 汇总 -----")
        print(f"  总关键帧: {len(self.keyframe_clouds)}, 有效帧: {len(self.keyframe_clouds) - zero_btc_frames}")
        print(f"  总BTC描述子: {total_btcs}, 零BTC帧数: {zero_btc_frames}")
        db_size = self.btc_manager.GetDatabaseSize() if hasattr(self.btc_manager, 'GetDatabaseSize') else 'N/A'
        print(f"  数据库大小: {db_size}")

        # ===== 步骤 3: 回环检测 + GICP 精化 + 验证 =====
        print("\n===== 步骤 3: 回环检测 =====")
        loop_constraints = []  # [(prev_idx, curr_idx, relative_pose_4x4, score), ...]
        step3_no_btc_frames = 0
        step3_search_count = 0
        step3_loop_count = 0
        odom_direct_verify_count = 0  # 新增：odom距离直接验证计数

        # 新增：odom距离阈值（小于此值直接做GICP，跳过BTC）
        odom_direct_threshold = self.odom_direct_threshold  # 从配置文件读取

        for i in range(len(self.keyframe_clouds)):
            cloud = self.keyframe_clouds[i]
            if len(cloud) < 10:
                print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: 点云点数不足 ({len(cloud)})，跳过")
                continue

            step3_search_count += 1

            # 提取当前帧odom位置（用于预过滤候选帧）
            pose = self.keyframe_poses[i]
            current_position = pose[:3, 3]  # 提取平移部分 (x, y, z)

            # ===== 新增策略: Odom距离直接验证 =====
            # 1. 先检查所有候选帧的odom距离
            odom_direct_candidates = []
            skip_near_num = self.skip_near_num  # 从配置文件读取
            for j in range(len(self.keyframe_clouds)):
                # 跳过当前帧和邻近帧（帧号差值必须 > skip_near_num）
                if j >= i:
                    continue  # 跳过当前帧和后面的帧
                frame_diff = i - j
                if frame_diff <= skip_near_num:
                    continue  # 跳过邻近帧（帧号差值太小）

                prev_pose = self.keyframe_poses[j]
                prev_position = prev_pose[:3, 3]
                odom_distance = np.linalg.norm(current_position - prev_position)

                # odom距离 < 3米：直接验证候选
                if odom_distance < odom_direct_threshold:
                    odom_direct_candidates.append((j, odom_distance))

            # 2. 对odom距离很近的候选帧，直接做GICP验证（跳过BTC）
            if odom_direct_candidates and self.use_gicp and HAS_GICP_OMP:
                for (candidate_id, odom_dist) in odom_direct_candidates:
                    print(f"  [Odom Direct] 帧 {i} <-> {candidate_id} odom距离 {odom_dist:.2f}m < {odom_direct_threshold}m，直接GICP验证")

                    # 计算相对位姿初值（从odom推算）
                    # keyframe_poses: world-from-body (odom姿态)
                    # relative_pose = inv(prev_pose) @ curr_pose → prev_frame←curr_frame 变换
                    curr_pose = self.keyframe_poses[i]
                    prev_pose = self.keyframe_poses[candidate_id]
                    relative_pose_matrix = np.linalg.inv(prev_pose) @ curr_pose
                    odom_init_t = np.linalg.norm(relative_pose_matrix[:3, 3])
                    print(f"  [Odom Direct] odom相对变换: t={odom_init_t:.3f}m, R(欧拉)={np.rad2deg(self._rot_to_euler(relative_pose_matrix[:3,:3]))}°")

                    # GICP验证（source=curr帧点云, target=prev帧点云, init=curr→prev变换）
                    gicp_result = gicp_align(
                        self.keyframe_clouds_ds[i],
                        self.keyframe_clouds_ds[candidate_id],
                        initial_guess=relative_pose_matrix,
                        config=self.gicp_config
                    )

                    # 无论验证是否成功，都保存融合点云供目视检查
                    if gicp_result and gicp_result.has_converged:
                        self._save_fused_cloud(i, candidate_id, gicp_result.transformation,
                                               relative_pose_matrix, gicp_result.fitness_score,
                                               gicp_result.overlap_ratio, 'odom_direct')

                    if (gicp_result and gicp_result.has_converged and
                            gicp_result.fitness_score < self.gicp_config.fitness_score_threshold):
                        # 检查退化方向
                        is_degenerate, deg_dir, eigvals = check_degeneracy(
                            self.keyframe_clouds_ds[i],
                            self.keyframe_clouds_ds[candidate_id],
                            gicp_result.transformation,
                            max_correspondence_distance=self.gicp_config.max_correspondence_distance
                        )

                        if not is_degenerate:
                            odom_direct_verify_count += 1
                            step3_loop_count += 1
                            print(f"  [Odom Direct SUCCESS] 回环验证成功! {candidate_id} <-> {i}, GICP fitness: {gicp_result.fitness_score:.4f}")

                            # 计算显示信息
                            T_delta = gicp_result.transformation
                            t_vec = T_delta[:3, 3]
                            roll, pitch, yaw = _extract_rpy(T_delta)

                            # 存储回环约束（统一格式）
                            loop_constraints.append((
                                candidate_id, i, gicp_result.transformation, 0.0, 'odom_direct'
                            ))
                            self.loop_pairs.append((
                                candidate_id, i,
                                f"{self.keyframe_indices[candidate_id]:06d}.pcd",
                                f"{self.keyframe_indices[i]:06d}.pcd",
                                0.0, gicp_result.fitness_score,
                                t_vec[0], t_vec[1], t_vec[2],
                                roll, pitch, yaw
                            ))
                            break  # 找到一个就跳出（避免重复验证）

            # ===== BTC匹配流程（正常流程） =====
            # C++ SearchLoop（使用已生成的BTC描述子）
            if cloud.shape[1] == 3:
                cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
            else:
                cloud_with_intensity = cloud

            # 生成当前帧BTC描述子（用于搜索）
            result = self.btc_manager.GenerateBtcDescs(cloud_with_intensity, i)
            num_btcs = result['num_btcs']

            if num_btcs == 0:
                step3_no_btc_frames += 1
                print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: BTC=0, 跳过")
                continue

            print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: 点云 {len(cloud)} pts, BTC={num_btcs}, 正在搜索回环...")

            # C++ SearchLoop（返回字典）
            result = self.btc_manager.SearchLoop(cloud_with_intensity, i, current_position)
            match_frame_id = result['match_frame_id']
            loop_score = result['match_score']
            candidates = result.get('candidate_frame_ids', [])

            # 诊断：SearchLoop返回的候选帧（从C++ candidate_matcher_vec收集）
            if match_frame_id == -1:
                if len(candidates) > 0:
                    print(f"  [BTC Diag] frame={i}, {len(candidates)} candidates, none passed verification")
                else:
                    print(f"  [BTC Diag] frame={i}, 0 candidates (no hash hits at all)")

            # 从字典中提取位姿
            if match_frame_id != -1:
                t = np.array(result['translation'])
                R = np.array(result['rotation'])
                relative_pose_matrix = np.eye(4)
                relative_pose_matrix[:3, :3] = R
                relative_pose_matrix[:3, 3] = t
            else:
                relative_pose_matrix = None

            # 回环处理（统一流程）
            if match_frame_id != -1 and relative_pose_matrix is not None:
                step3_loop_count += 1
                prev_idx = match_frame_id
                curr_idx = i
                print(f"  [BTC Loop] 检测到回环! {prev_idx} <-> {curr_idx}, score: {loop_score:.4f}")

                # GICP 精化 (对应 C++ performSCLoopClosure 中的 GICP 逻辑)
                gicp_success = False
                gicp_fitness = 0.0
                if self.use_gicp and HAS_GICP_OMP:
                    # ===== 新增策略: 使用odom位姿作为GICP初值（比BTC位姿更准确） =====
                    # 计算odom相对位姿（从关键帧位姿推算）
                    curr_pose = self.keyframe_poses[curr_idx]
                    prev_pose = self.keyframe_poses[prev_idx]
                    odom_relative_pose = np.linalg.inv(prev_pose) @ curr_pose

                    # 选择初值：优先使用odom位姿（更准确），BTC位姿作为备选
                    init_translation_btc = np.linalg.norm(relative_pose_matrix[:3, 3])
                    init_translation_odom = np.linalg.norm(odom_relative_pose[:3, 3])

                    # 如果odom平移 < BTC平移，说明BTC位姿不准，使用odom位姿
                    if init_translation_odom < init_translation_btc:
                        initial_guess = odom_relative_pose
                        print(f"  [GICP] 使用odom位姿作为初值 (odom={init_translation_odom:.2f}m < BTC={init_translation_btc:.2f}m)")
                    else:
                        initial_guess = relative_pose_matrix
                        print(f"  [GICP] 使用BTC位姿作为初值 (BTC={init_translation_btc:.2f}m <= odom={init_translation_odom:.2f}m)")

                    # 初始平移过大直接跳过（避免GICP崩溃）
                    init_translation = np.linalg.norm(initial_guess[:3, 3])
                    if init_translation > self.gicp_config.max_init_translation:  # 从yaml读取阈值
                        print(f"  [GICP] 初始平移 {init_translation:.1f}m > {self.gicp_config.max_init_translation}m，跳过")
                        continue

                    # 使用self.gicp_config（保留配置文件中的参数）
                    gicp_result = gicp_align(
                        self.keyframe_clouds_ds[curr_idx],
                        self.keyframe_clouds_ds[prev_idx],
                        initial_guess=initial_guess,
                        config=self.gicp_config
                    )
                    # 校验: Fitness Score
                    if (gicp_result and gicp_result.has_converged and
                            gicp_result.fitness_score < self.gicp_config.fitness_score_threshold):

                        # 校验: 退化方向检查
                        is_degenerate, deg_dir, eigvals = check_degeneracy(
                            self.keyframe_clouds_ds[curr_idx],
                            self.keyframe_clouds_ds[prev_idx],
                            gicp_result.transformation,
                            max_correspondence_distance=self.gicp_config.max_correspondence_distance
                        )

                        if not is_degenerate:
                            relative_pose_matrix = gicp_result.transformation
                            gicp_success = True
                            gicp_fitness = gicp_result.fitness_score
                            print(f"  [GICP] 精化成功! Fitness: {gicp_result.fitness_score:.4f}")

                            # 保存融合点云供目视检查
                            self._save_fused_cloud(curr_idx, prev_idx, gicp_result.transformation,
                                                   initial_guess, gicp_result.fitness_score,
                                                   gicp_result.overlap_ratio, 'btc_loop')
                        else:
                            print(f"  [GICP] 检测到退化方向，拒绝回环!")
                            continue

                    else:
                        fitness = gicp_result.fitness_score if gicp_result else float('inf')
                        print(f"  [GICP] 精化失败或分数过高 ({fitness:.4f} > {self.gicp_config.fitness_score_threshold})，拒绝回环!")
                        continue

                # 回环验证 (对应 C++ validateLoopClosure)
                if validate_loop_closure(
                        self.keyframe_poses[prev_idx],
                        self.keyframe_poses[curr_idx],
                        relative_pose_matrix,
                        max_loop_distance=self.max_loop_distance,
                        max_yaw_diff=self.max_yaw_diff):
                    # 找到对应的原始 scan 索引
                    merged_prev = self.keyframe_indices[prev_idx]
                    merged_curr = self.keyframe_indices[curr_idx]
                    if self.merge_n > 1 and self.merge_indices:
                        orig_prev_range = self.merge_indices[merged_prev]
                        orig_curr_range = self.merge_indices[merged_curr]
                        scan_prev_name = f"{orig_prev_range[0]:06d}-{orig_prev_range[-1]:06d}.pcd"
                        scan_curr_name = f"{orig_curr_range[0]:06d}-{orig_curr_range[-1]:06d}.pcd"
                        display_prev = f"{merged_prev} (orig {orig_prev_range[0]}-{orig_prev_range[-1]})"
                        display_curr = f"{merged_curr} (orig {orig_curr_range[0]}-{orig_curr_range[-1]})"
                    else:
                        scan_prev_name = f"{merged_prev:06d}.pcd"
                        scan_curr_name = f"{merged_curr:06d}.pcd"
                        display_prev = f"{merged_prev:06d}"
                        display_curr = f"{merged_curr:06d}"
                    # 计算相对位姿 (从 prev 到 curr)
                    T_delta = np.linalg.inv(self.keyframe_poses[prev_idx]) @ self.keyframe_poses[curr_idx]
                    t_vec = T_delta[:3, 3]
                    roll, pitch, yaw = _extract_rpy(T_delta)
                    print(f"\n  ===== 回环验证通过 =====")
                    print(f"  KeyFrame: {prev_idx} (Scan {display_prev}) <-> KeyFrame {curr_idx} (Scan {display_curr})")
                    print(f"  BTC score: {loop_score:.4f}, GICP fitness: {gicp_fitness:.4f}")
                    print(f"  PCD文件: Scans/{scan_prev_name} <-> Scans/{scan_curr_name}")
                    print(f"  相对平移: [{t_vec[0]:.3f}, {t_vec[1]:.3f}, {t_vec[2]:.3f}] m")
                    print(f"  相对旋转 (RPY): [{roll:.3f}, {pitch:.3f}, {yaw:.3f}] rad")
                    
                    # GICP失败时拒绝回环（不使用BTC原始位姿）
                    if not gicp_success:
                        print(f"  [WARN] GICP失败，拒绝此回环（BTC原始位姿可能错误）")
                        print(f"  =========================\n")
                        continue  # ← 跳过此回环
                    
                    print(f"  (使用GICP精化后的位姿)")
                    print(f"  =========================\n")

                    relative_pose_gtsam = matrix_to_gtsam_pose3(relative_pose_matrix)
                    loop_constraints.append((prev_idx, curr_idx, relative_pose_matrix, loop_score, 'btc'))
                    self.loop_pairs.append((
                        prev_idx, curr_idx,
                        scan_prev_name, scan_curr_name,
                        loop_score, gicp_fitness,
                        t_vec[0], t_vec[1], t_vec[2],
                        roll, pitch, yaw
                    ))
                    print(f"  [BTC Loop] 回环约束已添加: {prev_idx} <-> {curr_idx}")

        print(f"\n  ----- Step 3 汇总 -----")
        print(f"  搜索帧数: {step3_search_count}")
        print(f"  零BTC帧数: {step3_no_btc_frames}")
        print(f"  Odom直接验证: {odom_direct_verify_count} (odom距离<3m直接GICP)")
        print(f"  BTC检测回环: {step3_loop_count - odom_direct_verify_count}")
        print(f"  检测到回环: {step3_loop_count} (通过验证: {len(loop_constraints)})")
        print(f"  总计检测到 {len(loop_constraints)} 个回环")
        if loop_constraints:
            print(f"  ---- 通过验证的回环列表 ----")
            for i, constraint in enumerate(loop_constraints):
                # 统一格式: (prev_idx, curr_idx, rel_pose, score, loop_type)
                prev_idx, curr_idx, rel_pose, score, loop_type = constraint
                t = rel_pose[:3, 3]
                if loop_type == 'odom_direct':
                    print(f"  [{i+1}] 帧 {prev_idx} <-> {curr_idx} | odom_direct | "
                          f"平移: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
                else:
                    print(f"  [{i+1}] 帧 {prev_idx} <-> {curr_idx} | BTC | "
                          f"Score: {score:.4f} | 平移: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")

        # ===== 步骤 4: ISAM2 位姿图优化 (对应 C++ ISAM2) =====
        print("\n===== 步骤 4: ISAM2 位姿图优化 =====")
        optimized_poses = self._optimize_pose_graph(loop_constraints)

        # ===== 步骤 5: 保存结果 =====
        print("\n===== 步骤 5: 保存结果 =====")
        self._save_results(optimized_poses)

        return optimized_poses

    def _select_keyframes(self):
        """关键帧选择，与 C++ process_pg 中的逻辑一致"""
        self.keyframe_clouds = []
        self.keyframe_clouds_ds = []
        self.keyframe_poses = []
        self.keyframe_poses6d = []
        self.keyframe_indices = []  # 记录每个关键帧对应原始帧的索引

        translation_accumulated = float('inf')
        rotation_accumulated = float('inf')
        prev_pose6d = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(len(self.all_poses)):
            T = self.all_poses[i]
            pose6d = self._matrix_to_pose6d(T)

            # 计算增量
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

                # 保存原始点云 (BTC 使用) 和下采样点云 (GICP 使用)
                # 对应 C++ 中 keyframeLaserClouds 存储的是 downSizeFilterScancontext 后的完整帧
                cloud = self.all_scans[i]
                if len(cloud) > 0:
                    cloud_ds = down_sampling_voxel(cloud, self.scan_ds_size)
                else:
                    cloud_ds = cloud

                self.keyframe_clouds.append(cloud)       # BTC 用原始点云
                self.keyframe_clouds_ds.append(cloud_ds)  # GICP 用下采样点云
                self.keyframe_poses.append(T.copy())
                self.keyframe_poses6d.append(pose6d)
                self.keyframe_indices.append(i)  # 记录原始帧索引

            prev_pose6d = pose6d

    def _optimize_pose_graph(self, loop_constraints):
        """
        ISAM2 位姿图优化，与 C++ 版本一致:
          - prior noise: 1e-12
          - odom noise: [1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]
          - loop noise: Huber(1.345) + Diagonal(0.5)
        """
        if not HAS_GTSAM:
            print("[SKIP] GTSAM 不可用，跳过位姿图优化")
            return self.keyframe_poses

        if len(self.keyframe_poses) == 0:
            return self.keyframe_poses

        prior_noise, odom_noise, robust_loop_noise, robust_gps_noise = init_noises()

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

            if kf_idx_before == kf_idx_after:
                # 当前帧前后是同一个关键帧，直接用该关键帧的位姿
                T_opt = optimized_poses[kf_idx_before]
            else:
                # 线性插值权重（按原始帧索引距离）
                o_before = self.keyframe_indices[kf_idx_before]
                o_after = self.keyframe_indices[kf_idx_after]
                weight = (i - o_before) / max(o_after - o_before, 1)

                # 计算两个关键帧各自的校正量
                T_orig_before = self.keyframe_poses[kf_idx_before]
                T_orig_after = self.keyframe_poses[kf_idx_after]
                T_corr_before = np.linalg.inv(T_orig_before) @ optimized_poses[kf_idx_before]
                T_corr_after = np.linalg.inv(T_orig_after) @ optimized_poses[kf_idx_after]

                # 对平移做线性插值，旋转做球面线性插值（slerp）
                t_before = T_corr_before[:3, 3]
                t_after = T_corr_after[:3, 3]
                t_interp = t_before + weight * (t_after - t_before)

                R_before = T_corr_before[:3, :3]
                R_after = T_corr_after[:3, :3]
                R_interp = _interp_rotation_matrix(R_before, R_after, weight)

                T_corr = np.eye(4)
                T_corr[:3, :3] = R_interp
                T_corr[:3, 3] = t_interp

                T_opt = T_orig @ T_corr

            full_optimized_merged.append(T_opt)

        # 如果使用了多帧合并，将校正回推到原始帧
        if self.merge_n > 1 and self.original_poses is not None:
            print(f"  将优化结果回推到 {len(self.original_poses)} 个原始帧...")
            full_optimized = []
            for g, T_merged_opt in enumerate(full_optimized_merged):
                T_merged_orig = self.all_poses[g]  # 合并帧的原始 odom 位姿
                T_corr_g = np.linalg.inv(T_merged_orig) @ T_merged_opt

                for orig_idx in self.merge_indices[g]:
                    T_orig_i = self.original_poses[orig_idx]
                    T_opt_i = T_orig_i @ T_corr_g
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
        os.makedirs(fused_dir, exist_ok=True)

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

        _save_pcd_numpy_intensity(gicp_path, fused_pts_gicp, fused_int_gicp)
        _save_pcd_numpy_intensity(odom_path, fused_pts_odom, fused_int_odom)

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
                f.write(f"{R[0,0]:.10f} {R[0,1]:.10f} {R[0,2]:.10f} {t[0]:.10f} "
                        f"{R[1,0]:.10f} {R[1,1]:.10f} {R[1,2]:.10f} {t[1]:.10f} "
                        f"{R[2,0]:.10f} {R[2,1]:.10f} {R[2,2]:.10f} {t[2]:.10f}\n")

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
