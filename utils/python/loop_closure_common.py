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

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("[WARN] open3d 未安装，GICP 验证功能已禁用。安装: pip install open3d")

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False
    print("[WARN] gtsam 未安装，位姿图优化功能已禁用。安装: pip install gtsam")

from btc_common import BtcDescManager, load_config_setting, down_sampling_voxel


# ======================== GICP 配置 (对应 C++ GICPConfig) ========================

class GICPConfig:
    """对应 C++ GICPConfig，默认值与 gicp_config.yaml 一致"""

    def __init__(self):
        self.transformation_epsilon = 1e-6
        self.max_correspondence_distance = 30.0
        self.rotation_epsilon = 0.002
        self.k_correspondences = 20
        self.max_optimizer_iterations = 20
        self.gicp_epsilon = 0.01
        self.max_iterations = 100
        self.fitness_score_threshold = 0.5
        self.num_threads = 4


class GICPResult:
    """对应 C++ GICPResult"""

    def __init__(self):
        self.transformation = np.eye(4)
        self.has_converged = False
        self.fitness_score = float('inf')
        self.num_iterations = 0


def gicp_align(source_pts, target_pts, initial_guess=None, config=None):
    """
    对应 C++ GICPRegistration::align
    使用 Open3D GICP 进行点云配准。

    source_pts: Nx3 numpy array
    target_pts: Nx3 numpy array
    initial_guess: 4x4 numpy array
    config: GICPConfig
    返回: GICPResult
    """
    if not HAS_O3D:
        return None

    if config is None:
        config = GICPConfig()

    if initial_guess is None:
        initial_guess = np.eye(4)

    if len(source_pts) < 100 or len(target_pts) < 100:
        print("[GICP] 点云点数不足，跳过")
        return None

    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)
    target.points = o3d.utility.Vector3dVector(target_pts)

    # 估计法向量 (GICP 需要)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    result = GICPResult()

    try:
        reg_result = o3d.pipelines.registration.registration_generalized_icp(
            source, target,
            config.max_correspondence_distance,
            initial_guess,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
                epsilon=config.gicp_epsilon
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=config.transformation_epsilon,
                max_iteration=config.max_iterations
            )
        )

        result.transformation = reg_result.transformation
        result.has_converged = reg_result.fitness > 0
        result.fitness_score = reg_result.inlier_rmse
        result.num_iterations = config.max_iterations

        print(f"[GICP] Converged: {result.has_converged}, "
              f"Fitness score: {result.fitness_score:.4f}")

    except Exception as e:
        print(f"[GICP] 异常: {e}")
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
    if not HAS_O3D:
        return False, np.zeros(3), np.zeros(3)

    # 变换源点云
    ones = np.ones((len(source_pts), 1))
    source_homo = np.hstack([source_pts, ones])
    transformed = (transformation @ source_homo.T).T[:, :3]

    # 建立KD树查找对应点
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)
    kdtree = o3d.geometry.KDTreeFlann(target)

    # 收集残差向量
    residuals = []
    for pt in transformed:
        _, idxs, _ = kdtree.search_radius_vector_3d(pt, max_correspondence_distance)
        if len(idxs) > 0:
            # 找最近对应点
            target_pt = np.asarray(target.points)[idxs[0]]
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


# ======================== 位姿图优化 (对应 C++ GTSAM ISAM2) ========================

def init_noises():
    """
    对应 C++ initNoises
    返回 (prior_noise, odom_noise, robust_loop_noise, robust_gps_noise)
    """
    prior_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
    )
    odom_noise = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
    )
    loop_noise_score = 0.5
    robust_noise_vector = np.array([loop_noise_score] * 6)
    robust_loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Variances(robust_noise_vector)
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
                 use_gicp=True, scan_ds_size=0.4, use_cpp_btc=True,
                 debug_btc=False):
        """
        离线回环检测器

        默认使用C++ BTC实现（use_cpp_btc=True），算法与在线版本完全一致。
        若C++模块未安装，自动fallback到Python版本。

        参数:
            use_cpp_btc: bool, 默认True（使用C++ BTC）
            debug_btc: bool, 默认False（开启C++ BTC详细调试日志）
        """
        self.data_dir = data_dir or "/home/ywj/save_data/"
        self.use_cpp_btc = use_cpp_btc
        self.debug_btc = debug_btc

        # BTC 配置 - 默认使用C++实现，使用内置默认值
        if btc_config_file and os.path.exists(btc_config_file):
            # 指定配置文件时，加载覆盖默认值
            if use_cpp_btc:
                try:
                    import btc_cpp
                    self.btc_manager = btc_cpp.BtcDescManager(btc_config_file)
                    print(f"[BTC] 使用C++实现 (加载配置): {btc_config_file}")
                    if self.debug_btc:
                        self.btc_manager.SetDebugInfo(True)
                except ImportError:
                    print("[ERROR] C++ BTC模块未安装，fallback到Python版本")
                    self.btc_config = load_config_setting(btc_config_file)
                    self.btc_manager = BtcDescManager(self.btc_config)
                    self.use_cpp_btc = False
            else:
                self.btc_config = load_config_setting(btc_config_file)
                self.btc_manager = BtcDescManager(self.btc_config)
                print(f"[BTC] 使用Python实现 (加载配置): {btc_config_file}")
        else:
            # 未指定配置文件时，使用内置默认值（通用适配参数）
            if use_cpp_btc:
                try:
                    import btc_cpp
                    self.btc_manager = btc_cpp.BtcDescManager()  # 默认构造，使用内置默认值
                    print("[BTC] 使用C++实现 (内置默认配置)")
                    if self.debug_btc:
                        self.btc_manager.SetDebugInfo(True)
                except ImportError:
                    print("[ERROR] C++ BTC模块未安装，fallback到Python版本")
                    self.btc_manager = BtcDescManager()  # Python默认构造
                    self.use_cpp_btc = False
            else:
                self.btc_manager = BtcDescManager()  # Python默认构造，使用内置默认值
                print("[BTC] 使用Python实现 (内置默认配置)")
                
            self.btc_config = None

        # 启用调试日志
        if self.debug_btc and not self.use_cpp_btc:
            self.btc_manager.print_debug_info = True
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

    def run(self):
        """执行完整的离线回环流程"""
        if not hasattr(self, 'all_poses'):
            print("[ERROR] 请先调用 load_data()")
            return

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
                # C++ BTC和Python BTC的API差异处理
                if self.use_cpp_btc:
                    # C++ BTC: AddBtcDescs需要Nx4 numpy数组
                    # 将Nx3点云转换为Nx4（添加intensity）
                    if cloud.shape[1] == 3:
                        cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
                    else:
                        cloud_with_intensity = cloud
                    
                    # 添加到数据库
                    self.btc_manager.AddBtcDescs(cloud_with_intensity, i)
                    # 获取BTC数量（通过GenerateBtcDescs）
                    result = self.btc_manager.GenerateBtcDescs(cloud_with_intensity, i)
                    num_btcs = result['num_btcs']
                else:
                    # Python BTC: GenerateBtcDescs返回BTC列表
                    btcs_vec = self.btc_manager.GenerateBtcDescs(cloud, i)
                    self.btc_manager.AddBtcDescs(btcs_vec)
                    num_btcs = len(btcs_vec)

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
        if self.use_cpp_btc:
            db_size = self.btc_manager.GetDatabaseSize() if hasattr(self.btc_manager, 'GetDatabaseSize') else 'N/A'
            print(f"  数据库大小: {db_size}")
        else:
            print(f"  数据库位置数: {len(self.btc_manager.data_base)}")

        # ===== 步骤 3: 回环检测 + GICP 精化 + 验证 =====
        print("\n===== 步骤 3: 回环检测 =====")
        loop_constraints = []  # [(prev_idx, curr_idx, relative_pose_4x4, score), ...]
        step3_no_btc_frames = 0
        step3_search_count = 0
        step3_loop_count = 0

        for i in range(len(self.keyframe_clouds)):
            cloud = self.keyframe_clouds[i]
            if len(cloud) < 10:
                print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: 点云点数不足 ({len(cloud)})，跳过")
                continue

            step3_search_count += 1

            # C++ BTC和Python BTC的API差异处理
            if self.use_cpp_btc:
                # C++ BTC: SearchLoop直接搜索，返回字典
                if cloud.shape[1] == 3:
                    cloud_with_intensity = np.hstack([cloud, np.zeros((len(cloud), 1))])
                else:
                    cloud_with_intensity = cloud
                print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: 点云 {len(cloud)} pts, 正在搜索回环...")
                result = self.btc_manager.SearchLoop(cloud_with_intensity, i)
                match_frame_id = result['match_frame_id']
                loop_score = result['match_score']
                # 从字典中提取位姿
                if match_frame_id != -1:
                    t = np.array(result['translation'])
                    R = np.array(result['rotation'])
                    relative_pose_matrix = np.eye(4)
                    relative_pose_matrix[:3, :3] = R
                    relative_pose_matrix[:3, 3] = t
                    loop_transform = (t, R)
                    loop_std_pair = []  # C++ BTC不返回详细匹配对
                else:
                    loop_transform = None
                    loop_std_pair = None
                step3_no_btc_frames += 0  # C++内部已打印"No BTC descs!"
            else:
                # Python BTC: GenerateBtcDescs + SearchLoop
                btcs_vec = self.btc_manager.GenerateBtcDescs(cloud, i)
                if len(btcs_vec) == 0:
                    step3_no_btc_frames += 1
                print(f"  [Step3] 帧 {i}/{len(self.keyframe_clouds)}: 点云 {len(cloud)} pts, BTC: {len(btcs_vec)}, 正在搜索回环...")
                loop_result, loop_transform, loop_std_pair = self.btc_manager.SearchLoop(btcs_vec)
                match_frame_id, loop_score = loop_result
                if match_frame_id != -1:
                    t, R = loop_transform
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
                if self.use_gicp and HAS_O3D:
                    gicp_result = gicp_align(
                        self.keyframe_clouds_ds[curr_idx],
                        self.keyframe_clouds_ds[prev_idx],
                        initial_guess=relative_pose_matrix,
                        config=self.gicp_config
                    )

                    # Level 1 校验: Fitness Score
                    if (gicp_result and gicp_result.has_converged and
                            gicp_result.fitness_score < self.gicp_config.fitness_score_threshold):

                        # Level 2 校验: 退化方向检查（新增）
                        is_degenerate, deg_dir, eigvals = check_degeneracy(
                            self.keyframe_clouds_ds[curr_idx],
                            self.keyframe_clouds_ds[prev_idx],
                            gicp_result.transformation,
                            max_correspondence_distance=self.gicp_config.max_correspondence_distance
                        )

                        if not is_degenerate:
                            relative_pose_matrix = gicp_result.transformation
                            gicp_success = True
                            print(f"  [GICP] 精化成功! Fitness: {gicp_result.fitness_score:.4f}")
                        else:
                            print(f"  [GICP] 检测到退化方向，拒绝回环!")

                    else:
                        fitness = gicp_result.fitness_score if gicp_result else float('inf')
                        print(f"  [GICP] 精化失败或分数过高 ({fitness:.4f})，使用 BTC 结果")

                # 回环验证 (对应 C++ validateLoopClosure)
                if validate_loop_closure(
                        self.keyframe_poses[prev_idx],
                        self.keyframe_poses[curr_idx],
                        relative_pose_matrix):
                    relative_pose_gtsam = matrix_to_gtsam_pose3(relative_pose_matrix)
                    loop_constraints.append((prev_idx, curr_idx, relative_pose_matrix, loop_score))
                    self.loop_pairs.append((prev_idx, curr_idx, loop_score, 0.0))
                    print(f"  [BTC Loop] 回环约束已添加: {prev_idx} <-> {curr_idx}")

        print(f"\n  ----- Step 3 汇总 -----")
        print(f"  搜索帧数: {step3_search_count}")
        if not self.use_cpp_btc:
            print(f"  零BTC帧数: {step3_no_btc_frames}")
        print(f"  检测到回环: {step3_loop_count} (通过验证: {len(loop_constraints)})")
        print(f"  总计检测到 {len(loop_constraints)} 个回环")

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
        isam_params.relinearizeThreshold = 0.01
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
        for prev_idx, curr_idx, rel_pose_mat, score in loop_constraints:
            rel_gtsam = matrix_to_gtsam_pose3(rel_pose_mat)
            graph.add(gtsam.BetweenFactorPose3(prev_idx, curr_idx, rel_gtsam, robust_loop_noise))
            print(f"  添加回环因子: {prev_idx} <-> {curr_idx}")

        # 运行 ISAM2 优化
        isam.update(graph, initial)
        isam.update()

        current_estimate = isam.calculateEstimate()

        # 提取优化后的位姿
        optimized_poses = []
        for i in range(len(self.keyframe_poses)):
            if current_estimate.exists(i):
                pose3 = current_estimate.atPose3(i)
                optimized_poses.append(gtsam_pose3_to_matrix(pose3))
            else:
                optimized_poses.append(self.keyframe_poses[i].copy())

        print(f"  ISAM2 优化完成, {len(optimized_poses)} 个位姿")
        return optimized_poses

    def _save_results(self, optimized_poses):
        """保存结果"""
        data_dir = self.data_dir

        # 优化后的轨迹
        opt_file = os.path.join(data_dir, "optimized_poses.txt")
        self._save_poses_kitti(opt_file, optimized_poses)
        print(f"  优化轨迹已保存: {opt_file}")

        # 回环对
        pairs_file = os.path.join(data_dir, "loop_pairs.txt")
        with open(pairs_file, 'w') as f:
            f.write("# frame_a frame_b btc_score fitness_score\n")
            for a, b, s, fs in self.loop_pairs:
                f.write(f"{a} {b} {s:.6f} {fs:.6f}\n")
        print(f"  回环对已保存: {pairs_file}")

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
        if HAS_O3D:
            pcd = o3d.io.read_point_cloud(filepath)
            pts = np.asarray(pcd.points)
        else:
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
