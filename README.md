# SC-PGO (ROS2)

## 概述

SC-PGO 是一个基于位姿图优化（Pose-Graph Optimization）的 LiDAR 回环检测与建图系统，集成了 **BTC**（Binary Triangle Context）和 **ScanContext** 两种回环检测方法，并支持 **GICP_OMP** 精化配准。

本项目是 [SC-A-LOAM](https://github.com/gisbi-kim/SC-A-LOAM) 的 ROS2 移植版，针对 Mid360 固态激光雷达稀疏点云进行了大量适配优化，并增加了离线回环检测工具、Python C++ 绑定、可视化话题输出等新功能。

**核心流程：** 前端里程计（如 A-LOAM / LIO-SAM）→ 关键帧选取 → BTC/ScanContext 回环检测 → GICP 精化验证 → ISAM2 位姿图优化

## 功能特性

1. **BTC 回环检测（主要）** — 基于二进制三角形描述子，利用平面特征进行快速回环检索，适用于结构环境
2. **ScanContext 回环检测（备选）** — 经典全局描述子方法，支持结构化与非结构化环境
3. **GICP_OMP 精化配准** — 基于 pclomp 的 OpenMP 并行 GICP，提供精确相对位姿估计
4. **Odom Direct 验证** — 当里程计距离很近时，跳过 BTC 直接进行 GICP 验证，提升近距离回环召回率
5. **PGO 可视化话题** — 发布 `odom_keyframe_path`、`optimized_path`、`loop_match_markers` 话题，支持与 RViz 实时对比优化前后轨迹
6. **离线回环检测工具** — 提供完整的离线回环检测脚本，支持从 KITTI 格式位姿文件 + PCD 点云文件进行回环检测与位姿图优化
7. **C++ Python 绑定** — BTC、ScanContext、GICP 均通过 pybind11 暴露给 Python，离线脚本直接调用 C++ 实现，性能提升约 **100 倍**
8. **全参数可配置** — 所有回环检测、GICP、关键帧、回环验证参数均通过 YAML 配置文件管理
9. **Mid360 适配** — 针对固态激光雷达稀疏点云优化：多帧合并、放宽的 BTC 验证阈值、自适应参数配置
10. **模块化设计** — PGO 模块与前端里程计解耦，可接入任意里程计方法（A-LOAM、LIO-SAM、FAST-LIO 等）
11. **地图保存服务** — 提供 `save_map` ROS2 Service，支持手动触发保存合并后的点云地图和优化位姿，服务名称与保存文件名均可配置

## 依赖

- **ROS2 Humble / Foxy** — 核心通信框架
- **Ceres Solver** — 用于 A-LOAM 前端
- **GTSAM** — ISAM2 位姿图优化（`pip install gtsam`）
- **PCL** — 点云处理
- **OpenCV** — 配置文件解析
- **[ndt_omp_ros2](https://github.com/ypat999/ndt_omp_ros2.git)** — GICP_OMP 并行配准实现（工作空间内编译，见下文）
- **pybind11** — Python C++ 绑定
- **OpenMP** — 并行加速

安装系统依赖（Ubuntu 22.04）：

```bash
sudo apt install ros-humble-pcl-ros ros-humble-tf2-eigen libgoogle-glog-dev libgflags-dev
sudo apt install libpcl-dev libceres-dev
sudo apt install python3-pybind11 python3-dev
```

工作空间依赖：

`ndt_omp_ros2` 为编译和运行必需，须确保已包含在工作空间中：

| 包名 | 位置 / 来源 |
|---|---|
| **ndt_omp_ros2** | `https://github.com/ypat999/ndt_omp_ros2.git`，clone 到与本包同级目录 `../../src/ndt_omp_ros2`，提供 `pclomp::GICP_OMP` 实现 |

GTSAM 通过 pip 安装（`pip install gtsam`），无需放入工作空间。

## 编译

```bash
cd /home/ywj/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2
colcon build --packages-select sc_pgo_ros2
```

编译选项：

```bash
# 关闭 Python 绑定（嵌入式平台资源受限时）
colcon build --packages-select sc_pgo_ros2 \
  --cmake-args -DBUILD_PYTHON_BINDING=OFF
```

## 使用

### 在线模式（与前端里程计配合运行）

启动 PGO 节点，需同时运行前端里程计（如 LIO-SAM、A-LOAM）：

```bash
ros2 launch sc_pgo_ros2 sc_pgo.launch.py
```

launch 文件中需配置话题重映射，使 PGO 节点接收前端里程计的输出。

配置文件位于 `config/btc_config.yaml`，所有参数均可按需调整。

#### 保存地图服务

提供 `save_map` 服务用于手动触发保存地图：

```bash
ros2 service call /save_map std_srvs/srv/Trigger
```

保存内容：
- `<save_directory>/<map_filename>` — 合并后的点云地图（PCD 格式）
- `<save_directory>/optimized_poses_final.txt` — 优化后的位姿（KITTI 格式）

#### Launch 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `save_directory` | 保存目录路径 | `/home/ywj/save_data/` |
| `save_map_service_name` | 保存地图服务名称 | `save_map` |
| `map_filename` | 地图文件名 | `map.pcd` |

示例：

```bash
ros2 launch sc_pgo_ros2 sc_pgo.launch.py \
  save_directory:=/path/to/save/ \
  save_map_service_name:=my_save_map \
  map_filename:=my_map.pcd
```

#### 可视化话题

运行中会发布以下话题（可在 RViz 中查看）：

- `odom_keyframe_path` — PGO 输入的关键帧轨迹（优化前）
- `optimized_path` — PGO 优化后的轨迹
- `loop_match_markers` — 回环匹配点与连线（红色/绿色球体 + 黄色连线 + 标签）

### 离线模式

离线回环检测工具可对已记录的数据（KITTI 格式位姿 + PCD 点云文件）进行回环检测与位姿图优化：

```bash
cd src/SC_PGO_ROS2/utils/python
python3 offline_loop_closure.py /path/to/data/dir \
  --btc-config ../../config/btc_config.yaml \
  --merge-n 10
```

常用选项：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--btc-config` | YAML 配置文件路径 | 内置默认值 |
| `--keyframe-gap` | 关键帧间距阈值 (m) | 1.0 |
| `--merge-n` | 多帧合并数（Mid360 推荐 10） | 1 |
| `--no-gicp` | 禁用 GICP 精化 | false |
| `--debug-btc` | 开启 BTC 详细调试日志 | false |
| `--ros2` | 启用 ROS2 话题输出（可在 RViz 中查看） | false |

输出文件：

- `optimized_poses.txt` — KITTI 格式优化后轨迹
- `loop_pairs.txt` — 检测到的回环对列表
- `MergedScans/` — 合并后的点云（仅 `--merge-n > 1` 时）

示例：

```bash
# 基本用法（自动加载 C++ BTC）
python3 offline_loop_closure.py

# 自定义配置 + 多帧合并 + ROS2 可视化
python3 offline_loop_closure.py /home/ywj/save_data/ \
  --btc-config config/btc_config_outdoor.yaml \
  --merge-n 10 \
  --ros2

# 调整 GICP 阈值
python3 offline_loop_closure.py \
  --gicp-fitness-thres 0.5 \
  --gicp-max-dist 5.0
```

## 配置文件

`config/btc_config.yaml` 包含所有可配置参数，主要分为以下部分：

| 配置块 | 说明 |
|---|---|
| `scancontext` | 回环检测方法选择及 ScanContext 参数 |
| `voxel_size` | 体素化参数（第一阶段：平面检测） |
| `plane_merge_*` | 平面合并参数（第二阶段） |
| `proj_*` | 二进制描述子参数（第三阶段） |
| `descriptor_*` | BTC 三角形生成参数（第四阶段） |
| `skip_near_num` | 回环检索跳过的邻近帧数（第五阶段） |
| `candidate_num` | 候选帧数量阈值 |
| `ransac_*` | RANSAC 验证参数 |
| `icp_threshold` | BTC 平面几何验证分数阈值 |
| `loop_validation` | 回环验证参数（最大距离、偏航角差等） |
| `gicp` | GICP 精化参数 |
| `keyframe` | 关键帧选择参数 |

## 项目结构

```
src/SC_PGO_ROS2/
├── config/
│   └── btc_config.yaml          # 统一配置文件
├── include/
│   ├── btc/btc.h               # BTC 算法核心头文件
│   └── scancontext/             # ScanContext 头文件
├── src/
│   ├── laserPosegraphOptimization.cpp  # 在线 PGO 主节点
│   ├── btc.cpp                  # BTC 算法实现
│   ├── btc_python_binding.cpp   # BTC Python 绑定
│   ├── gicp_registration.cpp    # GICP 注册实现
│   ├── gicp_python_binding.cpp  # GICP Python 绑定
│   └── sc_python_binding.cpp    # ScanContext Python 绑定
├── utils/python/
│   ├── offline_loop_closure.py  # 离线回环检测工具
│   ├── loop_closure_common.py   # 公共回环检测逻辑（GICP、验证、优化）
│   ├── btc_common.py            # BTC 公共工具函数
│   ├── makeMergedMap.py         # 离线地图构建工具
│   ├── BTC_CPP_BINDING_BUILD.md # Python 绑定编译说明
│   └── README.md
├── launch/
│   └── sc_pgo.launch.py         # ROS2 launch 文件
└── config/
    └── btc_config.yaml          # 配置文件
```

## Python C++ 绑定

BTC、ScanContext、GICP 均已通过 pybind11 实现 Python C++ 绑定，离线脚本默认使用 C++ 实现。

性能对比：

| 实现 | BTC 生成 | 回环搜索 |
|---|---|---|
| C++ (pybind11) | ~10ms/帧 | ~5ms/帧 |
| Python | ~1.2s/帧 | ~0.5s/帧 |

详见 `utils/python/BTC_CPP_BINDING_BUILD.md`。

## 致谢

- [SC-A-LOAM](https://github.com/gisbi-kim/SC-A-LOAM) — 原始版本
- [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) — 前端里程计
- [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) — 位姿图优化框架参考
- [GTSAM](https://github.com/borglab/gtsam) — 因子图优化库
- [pclomp](https://github.com/ynpardb/ndt_omp) — GICP_OMP 并行配准

## 许可证

Apache License 2.0