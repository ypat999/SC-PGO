#!/usr/bin/env python3
"""
BTC 公共模块 — 仅保留数据结构和C++数据转换函数

Python版BTC实现已删除，全部使用C++ btc_cpp模块。
"""

import os
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
import yaml


# ======================== C++ BTC 数据转换函数 ========================

def convert_cpp_btc_to_python(btc_dict):
    """
    把 C++ BTC 模块返回的 dict 数据转换成 Python BTCDesc 对象。

    用于混合方案：C++ GenerateBtcDescs（快） + Python SearchLoop（有debug log）

    btc_dict 格式：
        {
            "triangle": [s1, s2, s3],
            "center": [x, y, z],
            "frame_number": int,
            "binary_A": {"location": [...], "summary": int, "normal": [...], "plane_id": int, "occupy_array": [bool, ...]},
            "binary_B": {...},
            "binary_C": {...}
        }
    """
    btc = BTCDesc()
    btc.triangle = np.array(btc_dict["triangle"], dtype=np.float64)
    btc.center = np.array(btc_dict["center"], dtype=np.float64)
    btc.frame_number = btc_dict["frame_number"]

    # BinaryDescriptor A
    btc.binary_A = BinaryDescriptor()
    btc.binary_A.location = np.array(btc_dict["binary_A"]["location"], dtype=np.float64)
    btc.binary_A.summary = btc_dict["binary_A"]["summary"]
    btc.binary_A.normal = np.array(btc_dict["binary_A"]["normal"], dtype=np.float64)
    btc.binary_A.plane_id = btc_dict["binary_A"]["plane_id"]
    btc.binary_A.occupy_array = list(btc_dict["binary_A"]["occupy_array"])  # ← 关键：二进制描述子数据

    # BinaryDescriptor B
    btc.binary_B = BinaryDescriptor()
    btc.binary_B.location = np.array(btc_dict["binary_B"]["location"], dtype=np.float64)
    btc.binary_B.summary = btc_dict["binary_B"]["summary"]
    btc.binary_B.normal = np.array(btc_dict["binary_B"]["normal"], dtype=np.float64)
    btc.binary_B.plane_id = btc_dict["binary_B"]["plane_id"]
    btc.binary_B.occupy_array = list(btc_dict["binary_B"]["occupy_array"])

    # BinaryDescriptor C
    btc.binary_C = BinaryDescriptor()
    btc.binary_C.location = np.array(btc_dict["binary_C"]["location"], dtype=np.float64)
    btc.binary_C.summary = btc_dict["binary_C"]["summary"]
    btc.binary_C.normal = np.array(btc_dict["binary_C"]["normal"], dtype=np.float64)
    btc.binary_C.plane_id = btc_dict["binary_C"]["plane_id"]
    btc.binary_C.occupy_array = list(btc_dict["binary_C"]["occupy_array"])

    return btc


def convert_cpp_btcs_list_to_python(btcs_data_list):
    """
    把 C++ GenerateBtcDescs 返回的 btcs_data list 转换成 Python BTCDesc list。

    btcs_data_list 格式：[btc_dict1, btc_dict2, ...]
    """
    return [convert_cpp_btc_to_python(btc_dict) for btc_dict in btcs_data_list]


# ======================== 数据结构 (对应 C++ struct) ========================

# P0-3: Union-Find (Disjoint Set) 数据结构，用于真正的连通域合并
class UnionFind:
    """对应 C++ UnionFind"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # 按秩合并
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def load_config_setting(config_file):
    """对应 C++ load_config_setting，从 yaml 文件加载参数"""
    cfg = ConfigSetting()
    if not os.path.exists(config_file):
        print(f"[WARN] 配置文件不存在: {config_file}，使用默认值")
        return cfg

    with open(config_file, 'r') as f:
        lines = f.readlines()
    # 跳过 OpenCV FileStorage 头 (%YAML:1.0)
    cleaned = ''.join(l for l in lines if not l.strip().startswith('%'))
    d = yaml.safe_load(cleaned)

    if d is None:
        return cfg

    # 字段映射: yaml key -> ConfigSetting attr
    mapping = {
        'useful_corner_num': 'useful_corner_num',
        'plane_merge_normal_thre': 'plane_merge_normal_thre',
        'plane_merge_dis_thre': 'plane_merge_dis_thre',
        'plane_detection_thre': 'plane_detection_thre',
        'voxel_size': 'voxel_size',
        'voxel_init_num': 'voxel_init_num',
        'proj_plane_num': 'proj_plane_num',
        'proj_image_resolution': 'proj_image_resolution',
        'proj_image_high_inc': 'proj_image_high_inc',
        'proj_dis_min': 'proj_dis_min',
        'proj_dis_max': 'proj_dis_max',
        'summary_min_thre': 'summary_min_thre',
        'line_filter_enable': 'line_filter_enable',
        'descriptor_near_num': 'descriptor_near_num',
        'descriptor_min_len': 'descriptor_min_len',
        'descriptor_max_len': 'descriptor_max_len',
        'max_constrait_dis': 'non_max_suppression_radius',
        'triangle_resolution': 'std_side_resolution',
        'skip_near_num': 'skip_near_num',
        'candidate_num': 'candidate_num',
        'rough_dis_threshold': 'rough_dis_threshold',
        'similarity_threshold': 'similarity_threshold',
        'icp_threshold': 'icp_threshold',
        'normal_threshold': 'normal_threshold',
        'dis_threshold': 'dis_threshold',
    }
    for yaml_key, attr_name in mapping.items():
        if yaml_key in d:
            setattr(cfg, attr_name, d[yaml_key])

    print(f"[BTC] 成功加载配置文件: {config_file}")
    return cfg


class BinaryDescriptor:
    """对应 C++ BinaryDescriptor"""

    def __init__(self):
        self.occupy_array = []   # list[bool]
        self.summary = 0         # unsigned char (int)
        self.location = np.zeros(3)  # Eigen::Vector3d
        self.normal = np.zeros(3)    # P0-1: 保存对应参考平面的法向量，避免generate_btc中的未定义行为
        self.plane_id = -1          # P2-1: 保存参考Plane ID，方便debug和可视化


class BTCDesc:
    """对应 C++ BTC"""

    def __init__(self):
        self.triangle = np.zeros(3)   # Eigen::Vector3d (scaled side lengths)
        self.angle = np.zeros(3)      # Eigen::Vector3d
        self.center = np.zeros(3)     # Eigen::Vector3d
        self.frame_number = 0         # unsigned short
        self.binary_A = BinaryDescriptor()
        self.binary_B = BinaryDescriptor()
        self.binary_C = BinaryDescriptor()


class Plane:
    """对应 C++ Plane"""

    def __init__(self):
        self.center = np.zeros(3)
        self.normal = np.zeros(3)
        self.covariance = np.zeros((3, 3))
        self.radius = 0.0
        self.min_eigen_value = 1.0
        self.d = 0.0
        self.id = 0
        self.sub_plane_num = 0
        self.points_size = 0
        self.is_plane = False


class BTCMatchList:
    """对应 C++ BTCMatchList"""

    def __init__(self):
        self.match_list = []      # list[(BTCDesc, BTCDesc)]
        self.match_id = (0, 0)    # (current_frame, match_frame)
        self.match_frame = 0
        self.mean_dis = 0.0


# ======================== 工具函数 ========================

def down_sampling_voxel(points, voxel_size):
    """
    对应 C++ down_sampling_voxel
    points: Nx3 numpy array
    返回体素下采样后的 Nx3 numpy array
    """
    if voxel_size < 0.01 or len(points) == 0:
        return points.copy()

    # 计算体素索引
    loc = points / voxel_size
    loc = np.floor(loc).astype(np.int64)

    voxel_map = {}
    for i in range(len(points)):
        key = (loc[i, 0], loc[i, 1], loc[i, 2])
        if key in voxel_map:
            entry = voxel_map[key]
            entry['sum'] += points[i]
            entry['count'] += 1
        else:
            voxel_map[key] = {'sum': points[i].copy(), 'count': 1}

    result = np.zeros((len(voxel_map), 3))
    for idx, entry in enumerate(voxel_map.values()):
        result[idx] = entry['sum'] / entry['count']
    return result


def binary_similarity(b1, b2):
    """
    对应 C++ binary_similarity
    b1, b2: BinaryDescriptor
    P1-2: 改用Jaccard相似度，增加惩罚项，避免descriptor太容易撞分
    """
    if len(b1.occupy_array) != len(b2.occupy_array):
        return 0.0

    match = 0      # 1-1 匹配（加分）
    mismatch = 0   # 1-0 或 0-1 不匹配（惩罚）

    for i in range(len(b1.occupy_array)):
        if b1.occupy_array[i] and b2.occupy_array[i]:
            match += 1
        elif b1.occupy_array[i] or b2.occupy_array[i]:
            mismatch += 1  # 惩罚项：一方occupied另一方不occupied
        # 0-0一致不计入，因为这是"共同空白"，不如"共同occupied"有价值

    # Jaccard相似度：match / (match + mismatch)
    # 当match=0时返回0，避免除零
    if match + mismatch == 0:
        return 0.0
    return match / (match + mismatch)


# ======================== OctoTree (对应 C++ OctoTree) ========================

class OctoTree:
    """对应 C++ OctoTree — 体素化平面检测"""

    def __init__(self, config):
        self.config = config
        self.voxel_points = []
        self.plane_ptr = Plane()
        self.layer = 0
        self.octo_state = 0
        self.init_octo = False

    def init_octo_tree(self):
        if len(self.voxel_points) > self.config.voxel_init_num:
            self._init_plane()

    def _init_plane(self):
        """对应 C++ OctoTree::init_plane"""
        pts = np.array(self.voxel_points)  # Nx3
        n = len(pts)
        self.plane_ptr.points_size = n
        self.plane_ptr.center = pts.mean(axis=0)
        centered = pts - self.plane_ptr.center
        self.plane_ptr.covariance = (centered.T @ centered) / n

        eigenvals, eigenvecs = LA.eigh(self.plane_ptr.covariance)
        # eigenvals 从小到大

        if eigenvals[0] < self.config.plane_detection_thre:
            self.plane_ptr.normal = eigenvecs[:, 0]
            self.plane_ptr.min_eigen_value = eigenvals[0]
            self.plane_ptr.radius = np.sqrt(eigenvals[2])
            self.plane_ptr.is_plane = True
            self.plane_ptr.d = -np.dot(self.plane_ptr.normal, self.plane_ptr.center)
        else:
            self.plane_ptr.is_plane = False

