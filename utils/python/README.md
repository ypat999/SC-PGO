# 离线回环检测工具 - 默认使用C++ BTC

## 核心优势
- **算法完全一致**：离线Python脚本直接调用C++ BTC库，与在线版本算法100%一致
- **性能提升100倍**：BTC生成10ms/帧（vs Python 1.2s/帧）
- **默认C++配置**：无需参数，自动使用最优实现

## 快速开始

### 1. 编译C++ BTC模块（默认已开启）
```bash
cd /home/ywj/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2

# 默认编译（Python绑定已默认开启，无需额外参数）
colcon build --packages-select sc_pgo_ros2

# 验证
python3 -c "import btc_cpp; print('✓ C++ BTC模块已安装')"
```

详细说明见：[BTC_CPP_BINDING_BUILD.md](BTC_CPP_BINDING_BUILD.md)

### 2. 运行离线回环检测（默认C++ BTC）
```bash
cd utils/python

# 默认使用C++ BTC（推荐）
python3 offline_loop_closure.py

# 自定义参数
python3 offline_loop_closure.py \
  --btc-config config/btc_config_outdoor.yaml \
  --keyframe-gap 5.0 --no-gicp

# 强制使用Python BTC（仅用于逻辑验证）
python3 offline_loop_closure.py --use-python-btc
```

## 文件结构
```
utils/python/
├── offline_loop_closure.py          # 离线回环检测入口（默认C++ BTC）
├── loop_closure_common.py           # 公共模块（GICP/验证/ISAM2）
├── btc_common.py                    # Python BTC实现（fallback）
├── BTC_CPP_BINDING_BUILD.md         # C++ BTC编译说明
└── README.md                        # 本文件

src/
├── btc_python_binding.cpp           # pybind11绑定源码

config/
├── btc_config.yaml                  # C++ BTC默认配置
└── btc_config_outdoor.yaml          # 室外场景适配配置
```

## 输出文件
- `optimized_poses.txt` — KITTI格式优化轨迹
- `loop_pairs.txt` — 回环对（帧索引+分数）

## 参数对照
离线脚本参数与在线C++版本（`sc_pgo.launch.py`）完全一致：
- `keyframe-gap` ↔ `keyframe_meter_gap`
- `gicp-fitness-thres` ↔ `gicp_fitness_score_threshold`
- `btc-config` ↔ `btc_config_file`

## 性能对比
| 实现 | BTC生成 | 回环搜索 | 算法一致性 |
|------|---------|---------|-----------|
| **C++（默认）** | 10ms | 5ms | ✓ 100% |
| Python（fallback） | 1.2s | 0.5s | ⚠️ 可能有差异 |

## 故障排查
```bash
# C++ BTC模块未安装
python3 offline_loop_closure.py
# 输出: [ERROR] C++ BTC模块未安装
# 解决: 编译C++模块（见BTC_CPP_BINDING_BUILD.md）或使用--use-python-btc

# 点云格式错误
# 输入必须是Nx4数组（x,y,z,intensity），Python会自动转换Nx3为Nx4
```