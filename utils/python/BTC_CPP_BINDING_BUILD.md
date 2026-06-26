# BTC C++ Python Binding 编译与使用说明

## 目标
将C++实现的BTC算法通过pybind11暴露给Python，离线脚本默认调用C++版本，保证算法完全一致并获得高性能。

## 默认配置（推荐）
离线脚本 `offline_loop_closure.py` 默认使用C++ BTC实现：
```bash
# 直接运行，自动使用C++ BTC（编译后）
python3 offline_loop_closure.py

# 强制使用Python版本（仅用于逻辑验证）
python3 offline_loop_closure.py --use-python-btc
```

## 依赖安装

### 1. 安装pybind11
```bash
# 方式1: 通过apt安装（推荐）
sudo apt install python3-pybind11

# 方式2: 通过pip安装
pip install pybind11
```

### 2. 安装Python开发包
```bash
sudo apt install python3-dev
```

## 编译步骤

### 1. 编译Python绑定（默认已开启）
```bash
cd /home/ywj/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2

# 默认编译（Python绑定已默认开启）
colcon build --packages-select sc_pgo_ros2

# 或者手动关闭Python绑定（嵌入式平台资源受限时）
colcon build --packages-select sc_pgo_ros2 \
  --cmake-args -DBUILD_PYTHON_BINDING=OFF
```

### 2. 安装Python模块
编译成功后，`btc_cpp`模块会自动安装到Python的site-packages目录。
验证安装：
```bash
python3 -c "import btc_cpp; print('BTC C++ module loaded:', btc_cpp.__file__)"
```

## 使用示例

### Python脚本调用C++ BTC
```python
import numpy as np
import btc_cpp

# 加载BTC配置
config_file = "/path/to/btc_config.yaml"
manager = btc_cpp.BtcDescManager(config_file)

# 查看配置参数
config = manager.GetConfig()
print(f"BTC配置: voxel_size={config['voxel_size']}, summary_min_thre={config['summary_min_thre']}")

# 加载点云（Nx4数组: x, y, z, intensity）
cloud = np.loadtxt("point_cloud.txt")  # 或从pcd文件读取
cloud = cloud[:, :4]  # 确保是Nx4格式

# 生成BTC描述子
result = manager.GenerateBtcDescs(cloud, frame_id=0)
print(f"生成 {result['num_btcs']} 个BTC")

# 添加到数据库
manager.AddBtcDescs(cloud, frame_id=0)

# 搜索回环
loop_result = manager.SearchLoop(cloud, frame_id=10)
if loop_result['match_frame_id'] != -1:
    print(f"检测到回环: 帧{loop_result['match_frame_id']}, 分数{loop_result['match_score']:.4f}")
    print(f"相对位姿: t={loop_result['translation']}, R={loop_result['rotation']}")
```

## 性能对比

| 实现 | BTC生成速度 | 回环搜索速度 |
|------|------------|-------------|
| C++ (pybind11) | ~10ms/帧 | ~5ms/帧 |
| Python移植版 | ~1.2s/帧 | ~0.5s/帧 |

**性能提升**: 约100倍

## 推荐的完整离线流程（默认C++ BTC）

离线脚本已默认集成C++ BTC，无需手动调用：

```bash
# 方式1: 使用离线脚本（推荐）
python3 offline_loop_closure.py

# 方式2: 自定义参数
python3 offline_loop_closure.py \
  --btc-config config/btc_config_outdoor.yaml \
  --keyframe-gap 10.0 \
  --no-gicp

# 方式3: 强制使用Python BTC（仅用于逻辑验证）
python3 offline_loop_closure.py --use-python-btc
```

---

## 手动调用C++ BTC示例（高级用法）

```python
import numpy as np
import btc_cpp

# 1. 加载配置
manager = btc_cpp.BtcDescManager("/path/to/btc_config_outdoor.yaml")

# 2. 加载关键帧点云
keyframes = [...]  # list of Nx4 numpy arrays (x,y,z,intensity)

# 3. 构建BTC数据库
for i, cloud in enumerate(keyframes):
    manager.AddBtcDescs(cloud, i)
    print(f"关键帧{i}: {manager.GetDatabaseSize()} BTC")

# 4. 搜索回环
loop_pairs = []
for i, cloud in enumerate(keyframes):
    result = manager.SearchLoop(cloud, i)
    if result['match_frame_id'] != -1:
        loop_pairs.append({
            'prev': result['match_frame_id'],
            'curr': i,
            'pose': result['translation'],
            'score': result['match_score']
        })

# 5. 位姿图优化 (使用GTSAM或其他优化器)
# ... (见loop_closure_common.py中的位姿图优化部分)
```

## 注意事项

1. **点云格式**: 输入必须是Nx4 numpy数组 (x, y, z, intensity)
2. **配置文件**: 使用C++版本的btc_config.yaml（OpenCV FileStorage格式）
3. **线程安全**: 单进程使用，多线程需要加锁
4. **内存管理**: 大场景需注意内存占用（数据库存储所有BTC）

## 编译失败排查

### 问题1: 找不到pybind11
```bash
# 检查pybind11是否安装
python3 -m pybind11 --version

# 手动指定pybind11路径
cmake .. -DBUILD_PYTHON_BINDING=ON \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
```

### 问题2: Python版本不匹配
```bash
# 检查Python版本
python3 --version

# 强制使用特定Python版本
cmake .. -DBUILD_PYTHON_BINDING=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3.10
```

### 问题3: PCL链接错误
```bash
# 检查PCL安装
pkg-config --modversion pcl_common

# 确保安装了PCL
sudo apt install libpcl-dev
```