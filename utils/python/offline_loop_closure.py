#!/usr/bin/env python3
"""
Super-LIO 离线回环检测工具 (BTC + GICP)

**只使用C++ BTC实现**：算法与在线版本完全一致，性能提升100倍。

读取 odom_poses.txt（KITTI 格式）和 Scans/*.pcd 文件，
执行与在线版本相同的流程:
  1. 关键帧选择 (keyframeMeterGap / keyframeRadGap)
  2. BTC 描述子生成 + 数据库构建 (C++ BtcDescManager)
  3. 回环检测 (SearchLoop: candidate_selector + candidate_verify)
  4. GICP 精化 (可选)
  5. 回环验证
  6. ISAM2 位姿图优化

输出文件：
  - optimized_poses.txt  : KITTI 格式优化后的轨迹
  - loop_pairs.txt       : 检测到的回环对

用法：
  python3 offline_loop_closure.py --btc-config ../../config/btc_config.yaml --merge-n 10 --debug-btc
"""

import os
import sys
import argparse

# 将当前目录和ROS2安装目录加入 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 添加ROS2 install目录到Python路径（用于加载btc_cpp模块）
ROS2_INSTALL_DIR = "/home/ywj/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/install/sc_pgo_ros2/lib/python3/dist-packages"
if os.path.exists(ROS2_INSTALL_DIR) and ROS2_INSTALL_DIR not in sys.path:
    sys.path.insert(0, ROS2_INSTALL_DIR)

# 注意: btc_cpp 的导入移到 _run_offline_loop_closure 内部，
# 避免 C++ 扩展共享库与 rclpy 冲突导致 Node() 创建时 SIGABRT。

from loop_closure_common import (
    OfflineLoopCloser,
    GICPConfig,
    HAS_BTC_CPP,
    load_unified_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线回环检测工具 — 使用C++ BTC确保与在线版本完全一致")

    parser.add_argument("data_dir", nargs="?", default="/home/ywj/save_data/",
                        help="数据目录 (默认: /home/ywj/save_data/)")

    parser.add_argument("--btc-config", type=str, default=None,
                        help="统一配置文件路径，包含BTC+GICP+关键帧+验证参数 (默认: 使用内置默认值)")

    # 关键帧参数 (对应 launch 文件中的 keyframe_meter_gap)
    parser.add_argument("--keyframe-gap", type=float, default=1.0,
                        help="关键帧间距阈值 (m), 默认 1.0 (可由配置文件覆盖)")
    parser.add_argument("--keyframe-deg-gap", type=float, default=10.0,
                        help="关键帧旋转阈值 (deg), 对应 keyframe_deg_gap (默认: 10.0)")

    # GICP 参数 (可由 --btc-config yaml 覆盖)
    parser.add_argument("--no-gicp", action="store_true",
                        help="禁用 GICP 精化（覆盖配置文件中的 gicp.enabled）")
    parser.add_argument("--gicp-fitness-thres", type=float, default=0.3,
                        help="GICP fitness score 阈值 (可由配置文件覆盖, 默认: 0.3)")
    parser.add_argument("--gicp-max-dist", type=float, default=3.0,
                        help="GICP 最大对应距离 (默认: 3.0)")
    parser.add_argument("--gicp-max-iter", type=int, default=32,
                        help="GICP 最大迭代次数 (默认: 32)")
    parser.add_argument("--gicp-epsilon", type=float, default=0.001,
                        help="GICP 协方差正则化/收敛精度 (默认: 0.001)")

    # 点云下采样（可由配置文件中的 gicp.scan_ds_size 覆盖）
    parser.add_argument("--scan-ds-size", type=float, default=0.1,
                        help="关键帧点云下采样体素大小 (m, 可由配置文件覆盖, 默认: 0.1)")

    # 调试参数
    parser.add_argument("--debug-btc", action="store_true",
                        help="开启C++ BTC详细调试日志（候选选择统计、平面验证等）")

    # 回环验证参数 (可由配置文件的 loop_validation 部分覆盖)
    parser.add_argument("--max-loop-distance", type=float, default=100.0,
                        help="最大回环距离 (m) (默认: 100.0)")
    parser.add_argument("--max-yaw-diff", type=float, default=None,
                        help="最大偏航角差 (rad), 默认从配置文件读取或 0.75π")
    parser.add_argument("--odom-direct-threshold", type=float, default=3.0,
                        help="Odom直接验证阈值 (m) - odom距离小于此值时直接GICP验证，跳过BTC (默认: 3.0)")
    parser.add_argument("--skip-near-num", type=int, default=5,
                        help="跳过邻近帧数 - 帧号差值<=此值时不检测回环 (默认: 5)")

    # 多帧合并参数 (针对点云稀疏的雷达如 Mid360)
    parser.add_argument("--merge-n", type=int, default=1,
                        help="每N帧合并为一帧，提升点云密度 (默认: 1, 不合并)。Mid360推荐10")

    # ROS2 可视化输出（用于在 RViz 中比较 PGO 效果）
    parser.add_argument("--ros2", action="store_true",
                        help="启用 ROS2 话题输出: odom_keyframe_path / optimized_path / loop_match_markers")
    parser.add_argument("--ros2-node-name", type=str, default="offline_loop_closure",
                        help="ROS2 节点名 (默认: offline_loop_closure)")
    parser.add_argument("--ros2-keep-alive", action="store_true",
                        help="处理完成后保持节点运行（便于在 RViz 中查看话题），按 Ctrl+C 退出")

    return parser.parse_args()


def main():
    args = parse_args()

    # 可选初始化 ROS2 节点（用于发布 odom_path / optimized_path / loop_match_markers 话题）
    ros_node = None
    rclpy_initialized = False
    if args.ros2:
        try:
            import rclpy
            from rclpy.node import Node
            rclpy.init()
            ros_node = Node(args.ros2_node_name)
            rclpy_initialized = True
            print(f"[ROS2] 节点已启动: {args.ros2_node_name}")
            print("[ROS2] 将发布话题:")
            print("  - odom_keyframe_path  (nav_msgs/Path)            PGO 输入轨迹")
            print("  - optimized_path      (nav_msgs/Path)            PGO 优化输出轨迹")
            print("  - loop_match_markers  (visualization_msgs/MarkerArray)  回环匹配点+连线")
        except ImportError:
            print("[WARN] rclpy 不可用，--ros2 选项已忽略（请先 source ROS2 环境）")
        except Exception as e:
            print(f"[WARN] ROS2 初始化失败: {e}")

    try:
        ret = _run_offline_loop_closure(args, ros_node)
    finally:
        if rclpy_initialized:
            if ros_node is not None and args.ros2_keep_alive:
                print("\n[ROS2] 处理完成，保持节点运行。按 Ctrl+C 退出...")
                try:
                    rclpy.spin(ros_node)
                except KeyboardInterrupt:
                    pass
            if ros_node is not None:
                ros_node.destroy_node()
            rclpy.shutdown()
    return ret


def _run_offline_loop_closure(args, ros_node):
    """实际的离线回环检测流程，便于在 ROS2 初始化/清理的 try-finally 中调用"""
    # 检查C++ BTC模块（由 loop_closure_common 管理导入，其导入顺序为 rclpy→btc_cpp）
    if not HAS_BTC_CPP:
        print("[ERROR] C++ BTC模块未安装，请先编译btc_cpp模块")
        print("[ERROR] 解决方案: 见 BTC_CPP_BINDING_BUILD.md")
        return 1

    # 从统一配置文件加载参数（CLI 可覆盖）
    btc_config_path = args.btc_config
    if btc_config_path and not os.path.exists(btc_config_path):
        # 尝试相对于 SCRIPT_DIR 解析（方便从不同目录运行）
        alt = os.path.join(SCRIPT_DIR, btc_config_path)
        if os.path.exists(alt):
            btc_config_path = alt
        elif not os.path.isabs(btc_config_path):
            # 尝试在包根目录查找 (config/ 在 src/SC_PGO_ROS2/ 下)
            pkg_root = os.path.dirname(SCRIPT_DIR)  # utils/python → ..
            alt2 = os.path.join(pkg_root, btc_config_path)
            if os.path.exists(alt2):
                btc_config_path = alt2

    if btc_config_path and os.path.exists(btc_config_path):
        cfg = load_unified_config(btc_config_path)
        btc_config_file = cfg['btc_config_path']
        gicp_config = cfg['gicp_config']
        use_gicp = cfg['use_gicp']
        keyframe_meter_gap = cfg['keyframe_meter_gap']
        keyframe_deg_gap = cfg['keyframe_deg_gap']
        scan_ds_size = cfg['gicp_config'].scan_ds_size
        max_loop_distance = cfg['max_loop_distance']
        max_yaw_diff = cfg['max_yaw_diff']
        odom_direct_threshold = cfg.get('odom_direct_threshold', 3.0)
        skip_near_num = cfg.get('skip_near_num', 5)  # 新增
    else:
        btc_config_file = args.btc_config
        gicp_config = GICPConfig()
        use_gicp = not args.no_gicp
        keyframe_meter_gap = args.keyframe_gap
        keyframe_deg_gap = args.keyframe_deg_gap
        scan_ds_size = args.scan_ds_size
        max_loop_distance = args.max_loop_distance
        max_yaw_diff = args.max_yaw_diff
        odom_direct_threshold = getattr(args, 'odom_direct_threshold', 3.0)
        skip_near_num = getattr(args, 'skip_near_num', 5)  # 新增

    # CLI 覆盖 GICP 参数（手动调参用）
    if args.gicp_fitness_thres != 0.3 or args.gicp_max_dist != 3.0 or \
       args.gicp_max_iter != 32 or args.gicp_epsilon != 0.001:
        gicp_config.fitness_score_threshold = args.gicp_fitness_thres
        gicp_config.max_correspondence_distance = args.gicp_max_dist
        gicp_config.max_iterations = args.gicp_max_iter
        gicp_config.transformation_epsilon = args.gicp_epsilon
        gicp_config.gicp_epsilon = args.gicp_epsilon
    if args.no_gicp:
        use_gicp = False
    if args.max_yaw_diff is not None:
        max_yaw_diff = args.max_yaw_diff
    # CLI 覆盖回环验证参数
    if args.max_loop_distance != 100.0:
        max_loop_distance = args.max_loop_distance
    if args.odom_direct_threshold != 3.0:
        odom_direct_threshold = args.odom_direct_threshold
    if hasattr(args, 'skip_near_num') and args.skip_near_num is not None:
        skip_near_num = args.skip_near_num

    # 创建离线回环检测器（只使用C++ BTC）
    closer = OfflineLoopCloser(
        data_dir=args.data_dir,
        btc_config_file=btc_config_path,  # 使用解析后的路径
        gicp_config=gicp_config,
        keyframe_meter_gap=keyframe_meter_gap,
        keyframe_deg_gap=keyframe_deg_gap,
        use_gicp=not args.no_gicp,
        scan_ds_size=scan_ds_size,
        debug_btc=args.debug_btc,
        max_loop_distance=max_loop_distance,
        max_yaw_diff=max_yaw_diff,
        odom_direct_threshold=odom_direct_threshold,
        skip_near_num=skip_near_num,  # 新增
        merge_n=args.merge_n,
        ros_node=ros_node
    )

    # 打印BTC配置参数
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
    try:
        optimized_poses = closer.run()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[ERROR] closer.run() 发生异常，跳过优化结果输出")
        return 1

    if optimized_poses is not None:
        print(f"\n[DONE] 优化完成, {len(optimized_poses)} 个关键帧位姿")
        print(f"       回环对: {len(closer.loop_pairs)}")
    else:
        print("\n[DONE] 回环检测完成，无优化结果")

    return 0


if __name__ == "__main__":
    sys.exit(main())