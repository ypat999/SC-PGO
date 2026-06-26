#!/usr/bin/env python3
"""
Super-LIO 离线回环检测工具 (BTC + GICP)

**默认使用C++ BTC实现**：算法与在线版本完全一致，性能提升100倍。
通过 --use-python-btc 参数可fallback到Python版本（仅用于逻辑验证）。

读取 odom_poses.txt（KITTI 格式）和 Scans/*.pcd 文件，
执行与在线版本相同的流程:
  1. 关键帧选择 (keyframeMeterGap / keyframeRadGap)
  2. BTC 描述子生成 + 数据库构建 (C++ BtcDescManager)
  3. 回环检测 (SearchLoop: candidate_selector + candidate_verify)
  4. GICP 精化 (可选, 对应 C++ GICPRegistration)
  5. 回环验证 (validateLoopClosure)
  6. ISAM2 位姿图优化

输出文件：
  - optimized_poses.txt  : KITTI 格式优化后的轨迹
  - loop_pairs.txt       : 检测到的回环对 (frame_a frame_b btc_score fitness_score)

用法：
  # 默认使用C++ BTC（自动）
  python3 offline_loop_closure.py

  # 强制使用Python BTC（仅用于逻辑验证，性能慢100倍）
  python3 offline_loop_closure.py --use-python-btc

  # 自定义参数
  python3 offline_loop_closure.py --btc-config config/btc_config_outdoor.yaml --no-gicp
"""

import os
import sys
import argparse
import numpy as np

# 将当前目录和ROS2安装目录加入 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 添加ROS2 install目录到Python路径（用于加载btc_cpp模块）
ROS2_INSTALL_DIR = "/home/ywj/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/install/sc_pgo_ros2/lib/python3/dist-packages"
if os.path.exists(ROS2_INSTALL_DIR) and ROS2_INSTALL_DIR not in sys.path:
    sys.path.insert(0, ROS2_INSTALL_DIR)

# 尝试加载C++ BTC模块
try:
    import btc_cpp
    HAS_CPP_BTC = True
    print("[INFO] C++ BTC模块已加载: btc_cpp")
except ImportError:
    HAS_CPP_BTC = False
    print("[WARN] C++ BTC模块未安装，将使用Python版本。编译方法见 BTC_CPP_BINDING_BUILD.md")

from loop_closure_common import (
    OfflineLoopCloser,
    GICPConfig,
    gicp_align,
    validate_loop_closure,
    init_noises,
    matrix_to_gtsam_pose3,
    gtsam_pose3_to_matrix,
)

from btc_common import (
    BtcDescManager,
    ConfigSetting,
    load_config_setting,
    down_sampling_voxel,
    binary_similarity,
    calc_triangle_dis,
    calc_binary_similarity,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线回环检测工具 — 使用C++ BTC确保与在线版本完全一致")

    parser.add_argument("data_dir", nargs="?", default="/home/ywj/save_data/",
                        help="数据目录 (默认: /home/ywj/save_data/)")

    # BTC实现选择（核心参数）
    parser.add_argument("--use-python-btc", action="store_true",
                        help="强制使用Python BTC实现（仅用于逻辑验证，性能慢100倍）")
    parser.add_argument("--btc-config", type=str, default=None,
                        help="BTC 配置文件路径 (默认: 使用内置通用适配参数)")

    # 关键帧参数 (对应 launch 文件中的 keyframe_meter_gap)
    parser.add_argument("--keyframe-gap", type=float, default=1.0,
                        help="关键帧间距阈值 (m), 对应 keyframe_meter_gap (默认: 5.0)")
    parser.add_argument("--keyframe-deg-gap", type=float, default=10.0,
                        help="关键帧旋转阈值 (deg), 对应 keyframe_deg_gap (默认: 10.0)")

    # GICP 参数 (对应 launch 文件中的 gicp_* 参数)
    parser.add_argument("--no-gicp", action="store_true",
                        help="禁用 GICP 精化")
    parser.add_argument("--gicp-fitness-thres", type=float, default=0.5,
                        help="GICP fitness score 阈值 (默认: 0.5)")
    parser.add_argument("--gicp-max-dist", type=float, default=30.0,
                        help="GICP 最大对应距离 (默认: 30.0)")
    parser.add_argument("--gicp-max-iter", type=int, default=100,
                        help="GICP 最大迭代次数 (默认: 100)")
    parser.add_argument("--gicp-epsilon", type=float, default=1e-6,
                        help="GICP 收敛阈值 (默认: 1e-6)")

    # 点云下采样
    parser.add_argument("--scan-ds-size", type=float, default=0.1,
                        help="关键帧点云下采样体素大小 (m), 对应 downSizeFilterScancontext (默认: 0.4)")

    # 调试参数
    parser.add_argument("--debug-btc", action="store_true",
                        help="开启C++ BTC详细调试日志（平面检测、合并率、描述子数等）")

    # 回环验证参数 (对应 C++ validateLoopClosure)
    parser.add_argument("--max-loop-distance", type=float, default=100.0,
                        help="最大回环距离 (m) (默认: 100.0)")
    parser.add_argument("--max-yaw-diff", type=float, default=None,
                        help="最大偏航角差 (rad), 默认 0.75π")

    return parser.parse_args()


def main():
    args = parse_args()

    # 确定BTC实现：默认C++，--use-python-btc强制Python
    use_cpp_btc = not args.use_python_btc

    # 检查C++ BTC模块
    if use_cpp_btc and not HAS_CPP_BTC:
        print("[ERROR] C++ BTC模块未安装，无法使用默认配置。")
        print("[ERROR] 解决方案:")
        print("[ERROR]   1. 编译C++ BTC: 见 BTC_CPP_BINDING_BUILD.md")
        print("[ERROR]   2. 或使用Python版本: --use-python-btc")
        return 1

    # 构建 GICP 配置 (与 launch 文件参数一致)
    gicp_config = GICPConfig()
    gicp_config.fitness_score_threshold = args.gicp_fitness_thres
    gicp_config.max_correspondence_distance = args.gicp_max_dist
    gicp_config.max_iterations = args.gicp_max_iter
    gicp_config.transformation_epsilon = args.gicp_epsilon

    # 创建离线回环检测器 (默认使用C++ BTC)
    closer = OfflineLoopCloser(
        data_dir=args.data_dir,
        btc_config_file=args.btc_config,
        gicp_config=gicp_config,
        keyframe_meter_gap=args.keyframe_gap,
        keyframe_deg_gap=args.keyframe_deg_gap,
        use_gicp=not args.no_gicp,
        scan_ds_size=args.scan_ds_size,
        use_cpp_btc=use_cpp_btc,  # 默认True
        debug_btc=args.debug_btc,
    )

    # 打印BTC配置参数
    if use_cpp_btc:
        try:
            config = closer.btc_manager.GetConfig()
            print("\n===== BTC 配置参数 =====")
            for k, v in config.items():
                print(f"  {k}: {v}")
        except:
            pass

    # 加载数据
    if not closer.load_data():
        print("[ERROR] 数据加载失败")
        return 1

    # 执行回环检测 + 优化
    optimized_poses = closer.run()

    if optimized_poses is not None:
        print(f"\n[DONE] 优化完成, {len(optimized_poses)} 个关键帧位姿")
        print(f"       回环对: {len(closer.loop_pairs)}")
    else:
        print("\n[DONE] 回环检测完成，无优化结果")

    return 0


if __name__ == "__main__":
    sys.exit(main())