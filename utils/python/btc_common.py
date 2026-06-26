#!/usr/bin/env python3
"""
BTC 公共模块 — Python 忠实移植 C++ BtcDescManager

与 C++ 源码 (btc.h / btc.cpp) 算法完全对齐，供离线工具和在线推理共用。
"""

import os
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
import yaml

try:
    from scipy.spatial import KDTree as ScipyKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


# ======================== 数据结构 (对应 C++ struct) ========================

class ConfigSetting:
    """对应 C++ ConfigSetting，默认值与 btc_config.yaml 一致（通用适配默认值 - 优化版）"""

    def __init__(self):
        # ========== 平面检测参数（通用适配默认值 - 优化版）==========
        self.cloud_ds_size = 0.25

        self.plane_detection_thre = 0.01
        self.voxel_size = 0.5  # 降低以增加体素密度（原1.0导致体素内点数不足）
        self.voxel_init_num = 3  # 降低以适应稀疏点云（原5阈值过高）

        # ========== 平面合并参数 ==========
        self.plane_merge_normal_thre = 0.1
        self.plane_merge_dis_thre = 0.3
        self.plane_merge_search_radius = 2.0  # P1-3: KDTree邻域搜索半径，替代全局O(N²)比较

        # ========== 二进制描述子参数 ==========
        self.proj_plane_num = 5  # 增加多方向投影
        self.proj_image_resolution = 0.5
        self.proj_image_high_inc = 0.5
        self.proj_dis_min = 0.0
        self.proj_dis_max = 5.0
        self.summary_min_thre = 3  # 降低以适应单层扫描
        self.line_filter_enable = 0

        # ========== BTC 生成参数 ==========
        self.descriptor_near_num = 5  # 降低允许较少描述子生成三角形
        self.descriptor_min_len = 1.0
        self.descriptor_max_len = 10.0
        self.non_max_suppression_radius = 3.0
        self.nms_score_margin = 1.0  # P1-4: NMS score margin，避免大量descriptor被误删
        self.std_side_resolution = 0.2

        self.useful_corner_num = 20  # 降低

        # ========== 回环检测参数 ==========
        self.skip_near_num = 5  # 降低因静态场景帧间距离极小
        self.candidate_num = 50
        self.sub_frame_num = 10

        # ========== 验证参数 ==========
        self.rough_dis_threshold = 0.03
        self.similarity_threshold = 0.7
        self.icp_threshold = 0.5
        self.normal_threshold = 0.1
        self.dis_threshold = 0.3
        self.icp_min_match_num = 20  # P2-2: ICP最小有效匹配数，避免退化优化
        self.triangle_resolution = 0.2


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


# ======================== BtcDescManager (对应 C++ BtcDescManager) ========================

class BtcDescManager:
    """
    Python 忠实移植 C++ BtcDescManager。
    核心方法: GenerateBtcDescs, SearchLoop, AddBtcDescs
    """

    def __init__(self, config=None):
        if config is None:
            config = ConfigSetting()
        self.config = config
        self.data_base = {}         # BTC_LOC -> [BTCDesc, ...]
        self.key_cloud_vec = []     # 保留关键帧点云(用于GICP)
        self.history_binary_list = []
        self.plane_cloud_vec = []   # 平面点云(中心+法向量)
        self.print_debug_info = False

    # ---------- 公共 API (与 C++ 一致) ----------

    def GenerateBtcDescs(self, input_cloud, frame_id):
        """
        对应 C++ BtcDescManager::GenerateBtcDescs
        input_cloud: Nx3 numpy array (点云)
        frame_id: int
        返回: [BTCDesc, ...]
        P2-4: 增加大量调试统计信息
        """
        btcs_vec = []

        if self.print_debug_info:
            print(f"[GenerateBtcDescs] Frame {frame_id} - input_cloud size: {len(input_cloud)}")

        # 1. 体素化 + 平面检测
        voxel_map = self._init_voxel_map(input_cloud)

        if self.print_debug_info:
            print(f"[GenerateBtcDescs] Voxel map size: {len(voxel_map)}")

        # 2. 提取平面点云 (中心+法向量)
        plane_cloud = self._get_plane(voxel_map)
        if self.print_debug_info:
            print(f"[GenerateBtcDescs] Plane cloud size: {len(plane_cloud)}")
        self.plane_cloud_vec.append(plane_cloud)

        # P2-4: 增加调试统计信息 - 原始平面数量
        original_plane_num = 0
        for tree in voxel_map.values():
            if tree.plane_ptr.is_plane:
                original_plane_num += 1

        # 3. 获取投影平面 & 合并
        proj_plane_list = self._get_project_plane(voxel_map)
        if len(proj_plane_list) == 0:
            single_plane = Plane()
            single_plane.normal = np.array([0, 0, 1])
            single_plane.center = input_cloud[0]
            merge_plane_list = [single_plane]
        else:
            proj_plane_list.sort(key=lambda p: p.points_size, reverse=True)
            merge_plane_list = self._merge_plane(proj_plane_list)
            merge_plane_list.sort(key=lambda p: p.points_size, reverse=True)

        # P2-4: 增加调试统计信息 - 合并后平面数量和合并率
        merged_plane_num = len(merge_plane_list)
        if self.print_debug_info:
            # Python使用条件表达式而不是?:三元运算符
            merge_ratio = (1.0 - float(merged_plane_num) / float(original_plane_num)) if (original_plane_num > 0) else 0.0
            print(f"[GenerateBtcDescs] Original planes: {original_plane_num}, "
                  f"merged planes: {merged_plane_num}, merge ratio: {merge_ratio * 100:.1f}%")

        # 4. 提取二进制描述子
        binary_list = self._binary_extractor(merge_plane_list, input_cloud)
        self.history_binary_list.append(binary_list)
        if self.print_debug_info:
            print(f"[GenerateBtcDescs] Binary descriptors: {len(binary_list)}")

        # 5. 生成 BTC 三角形
        btcs_vec = self._generate_btc(binary_list, frame_id)
        if self.print_debug_info:
            print(f"[GenerateBtcDescs] BTC descriptors: {len(btcs_vec)}")

        return btcs_vec

    def SearchLoop(self, btcs_vec):
        """
        对应 C++ BtcDescManager::SearchLoop
        返回: (loop_result, loop_transform, loop_std_pair)
            loop_result: (match_frame_id, score) 或 (-1, 0)
            loop_transform: (t_vec3, R_mat3x3)
            loop_std_pair: [(BTCDesc, BTCDesc), ...]
        """
        if len(btcs_vec) == 0:
            print("[BTC] No BTC descs!")
            return (-1, 0.0), (np.zeros(3), np.eye(3)), []

        # 1. 候选选择
        candidate_matcher_vec = self._candidate_selector(btcs_vec)

        # 2. 候选验证
        best_score = 0.0
        best_candidate_id = -1
        best_transform = (np.zeros(3), np.eye(3))
        best_sucess_match_vec = []

        for candidate in candidate_matcher_vec:
            verify_score, relative_pose, sucess_match_vec = self._candidate_verify(candidate)
            if self.print_debug_info:
                print(f"[Retrieval] try frame: {candidate.match_id[1]}, "
                      f"rough size: {len(candidate.match_list)}, score: {verify_score}")

            if verify_score > best_score:
                best_score = verify_score
                best_candidate_id = candidate.match_id[1]
                best_transform = relative_pose
                best_sucess_match_vec = sucess_match_vec

        if self.print_debug_info:
            print(f"[Retrieval] best candidate: {best_candidate_id}, score: {best_score}")

        if best_score > self.config.icp_threshold:
            return (best_candidate_id, best_score), best_transform, best_sucess_match_vec
        else:
            return (-1, 0.0), (np.zeros(3), np.eye(3)), []

    def AddBtcDescs(self, btcs_vec):
        """
        对应 C++ BtcDescManager::AddBtcDescs
        P1-1: Triangle Hash改为可配置量化，使用std_side_resolution_作为分辨率
        P2-4: 增加调试日志
        """
        if self.print_debug_info:
            print(f"[AddBtcDescs] Adding {len(btcs_vec)} BTC descriptors to database")

        for single_std in btcs_vec:
            # P1-1: 使用可配置分辨率进行量化，避免bucket太粗
            resolution = self.config.std_side_resolution
            px = int(single_std.triangle[0] / resolution + 0.5)
            py = int(single_std.triangle[1] / resolution + 0.5)
            pz = int(single_std.triangle[2] / resolution + 0.5)
            position = (px, py, pz)

            if self.print_debug_info:
                print(f"[AddBtcDescs] BTC triangle: {single_std.triangle}, hash position: {position}")

            if position in self.data_base:
                self.data_base[position].append(single_std)
            else:
                self.data_base[position] = [single_std]

        if self.print_debug_info:
            print(f"[AddBtcDescs] Database size: {len(self.data_base)} positions")

    # ---------- 内部方法 ----------

    def _init_voxel_map(self, input_cloud):
        """对应 C++ BtcDescManager::init_voxel_map"""
        voxel_map = {}
        for i in range(len(input_cloud)):
            p = input_cloud[i]
            loc = p / self.config.voxel_size
            loc_xyz = np.floor(loc).astype(np.int64)
            key = (loc_xyz[0], loc_xyz[1], loc_xyz[2])
            if key in voxel_map:
                voxel_map[key].voxel_points.append(p.copy())
            else:
                tree = OctoTree(self.config)
                tree.voxel_points.append(p.copy())
                voxel_map[key] = tree

        # 初始化所有 OctoTree
        for tree in voxel_map.values():
            tree.init_octo_tree()

        return voxel_map

    def _get_plane(self, voxel_map):
        """对应 C++ BtcDescManager::get_plane — 返回 [(center, normal), ...]"""
        plane_cloud = []
        for tree in voxel_map.values():
            if tree.plane_ptr.is_plane:
                plane_cloud.append((tree.plane_ptr.center.copy(),
                                    tree.plane_ptr.normal.copy()))
        return plane_cloud

    def _get_project_plane(self, voxel_map):
        """对应 C++ BtcDescManager::get_project_plane"""
        origin_list = []
        for tree in voxel_map.values():
            if tree.plane_ptr.is_plane:
                origin_list.append(tree.plane_ptr)

        for p in origin_list:
            p.id = 0

        current_id = 1
        # 反向遍历合并平面 ID
        for i in range(len(origin_list) - 1, 0, -1):
            for j in range(0, i):
                pi = origin_list[i]
                pj = origin_list[j]
                normal_diff = pi.normal - pj.normal
                normal_add = pi.normal + pj.normal
                dis1 = abs(np.dot(pi.normal, pj.center) + pi.d)
                dis2 = abs(np.dot(pj.normal, pi.center) + pj.d)
                if (LA.norm(normal_diff) < self.config.plane_merge_normal_thre or
                        LA.norm(normal_add) < self.config.plane_merge_normal_thre):
                    if dis1 < self.config.plane_merge_dis_thre and dis2 < self.config.plane_merge_dis_thre:
                        if pi.id == 0 and pj.id == 0:
                            pi.id = current_id
                            pj.id = current_id
                            current_id += 1
                        elif pi.id == 0 and pj.id != 0:
                            pi.id = pj.id
                        elif pi.id != 0 and pj.id == 0:
                            pj.id = pi.id

        merge_list = []
        merge_flag = []

        for i in range(len(origin_list)):
            if origin_list[i].id in merge_flag:
                continue
            if origin_list[i].id == 0:
                continue
            merge_plane = Plane()
            # 深拷贝
            merge_plane.center = origin_list[i].center.copy()
            merge_plane.normal = origin_list[i].normal.copy()
            merge_plane.covariance = origin_list[i].covariance.copy()
            merge_plane.d = origin_list[i].d
            merge_plane.radius = origin_list[i].radius
            merge_plane.min_eigen_value = origin_list[i].min_eigen_value
            merge_plane.points_size = origin_list[i].points_size
            merge_plane.sub_plane_num = origin_list[i].sub_plane_num
            merge_plane.id = origin_list[i].id
            merge_plane.is_plane = origin_list[i].is_plane

            is_merge = False
            for j in range(len(origin_list)):
                if i == j:
                    continue
                if origin_list[j].id == origin_list[i].id:
                    is_merge = True
                    P_PT1 = (merge_plane.covariance + np.outer(merge_plane.center, merge_plane.center)) * merge_plane.points_size
                    P_PT2 = (origin_list[j].covariance + np.outer(origin_list[j].center, origin_list[j].center)) * origin_list[j].points_size
                    total_pts = merge_plane.points_size + origin_list[j].points_size
                    merge_center = (merge_plane.center * merge_plane.points_size + origin_list[j].center * origin_list[j].points_size) / total_pts
                    merge_cov = (P_PT1 + P_PT2) / total_pts - np.outer(merge_center, merge_center)

                    merge_plane.covariance = merge_cov
                    merge_plane.center = merge_center
                    merge_plane.points_size = total_pts
                    merge_plane.sub_plane_num += 1

                    # 重新计算特征值
                    eigenvals, eigenvecs = LA.eigh(merge_cov)
                    merge_plane.normal = eigenvecs[:, 0]
                    merge_plane.radius = np.sqrt(eigenvals[2])
                    merge_plane.d = -np.dot(merge_plane.normal, merge_plane.center)

            if is_merge:
                merge_flag.append(merge_plane.id)
                merge_list.append(merge_plane)

        return merge_list

    def _merge_plane(self, origin_list):
        """
        对应 C++ BtcDescManager::merge_plane
        P0-3: 使用Union-Find进行真正的连通域合并
        P1-3: 使用KDTree降低复杂度
        P2-4: 增加大量调试日志
        """
        if len(origin_list) == 1:
            return list(origin_list)

        if self.print_debug_info:
            print(f"[MergePlane] Start - origin_list size: {len(origin_list)}")

        # P0-3: 使用Union-Find进行真正的连通域合并
        # P1-3: 使用KDTree降低复杂度，只比较radius内的平面
        uf = UnionFind(len(origin_list))

        # 构建KDTree，使用平面中心点
        plane_centers = np.array([p.center for p in origin_list])

        if self.print_debug_info:
            print(f"[MergePlane] Building KDTree for {len(plane_centers)} plane centers")

        # 使用scipy KDTree或自己实现的版本
        if HAS_SCIPY:
            kd_tree = ScipyKDTree(plane_centers)
        else:
            # 如果没有scipy，使用简单的半径搜索（性能较低）
            kd_tree = None

        search_radius = self.config.plane_merge_search_radius

        if self.print_debug_info:
            print(f"[MergePlane] Using search_radius: {search_radius}")

        # 第一遍遍历：使用KDTree找到邻域平面，满足条件则Union
        for i in range(len(origin_list)):
            if kd_tree is not None:
                # 使用KDTree查询邻域
                neighbors = kd_tree.query_ball_point(origin_list[i].center, search_radius)
                for j in neighbors:
                    if j <= i:  # 避免重复比较
                        continue
            else:
                # 没有KDTree时使用简单搜索
                neighbors = []
                for j in range(i + 1, len(origin_list)):
                    if LA.norm(origin_list[i].center - origin_list[j].center) < search_radius:
                        neighbors.append(j)

            for j in neighbors:
                pi = origin_list[i]
                pj = origin_list[j]
                normal_diff = pi.normal - pj.normal
                normal_add = pi.normal + pj.normal
                dis1 = abs(np.dot(pi.normal, pj.center) + pi.d)
                dis2 = abs(np.dot(pj.normal, pi.center) + pj.d)

                # Python语法要求：跨行if语句需要用括号包裹整个表达式
                if ((LA.norm(normal_diff) < self.config.plane_merge_normal_thre or
                        LA.norm(normal_add) < self.config.plane_merge_normal_thre) and
                        dis1 < self.config.plane_merge_dis_thre and
                        dis2 < self.config.plane_merge_dis_thre):
                    uf.union(i, j)  # 合并满足条件的平面

        # 第二遍遍历：按root聚类
        clusters = {}
        for i in range(len(origin_list)):
            root = uf.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        if self.print_debug_info:
            print(f"[MergePlane] Found {len(clusters)} clusters")
            # for root, indices in clusters.items():
            #     print(f"[MergePlane] Cluster {root}: {len(indices)} planes")

        # 对每个聚类重新计算合并后的Plane
        merge_plane_list = []
        for root, indices in clusters.items():
            if len(indices) == 1:
                # 单独的平面，不合并
                merge_plane_list.append(origin_list[indices[0]])
                origin_list[indices[0]].id = 0
                # if self.print_debug_info:
                #     print(f"[MergePlane] Single plane {indices[0]} - no merge")
            else:
                # 多个平面需要合并
                merge_plane = Plane()
                merge_plane = origin_list[indices[0]]

                P_PT_sum = (merge_plane.covariance + np.outer(merge_plane.center, merge_plane.center)) * merge_plane.points_size
                center_sum = merge_plane.center * merge_plane.points_size
                total_points = merge_plane.points_size

                for idx in range(1, len(indices)):
                    j = indices[idx]
                    P_PT_sum += (origin_list[j].covariance + np.outer(origin_list[j].center, origin_list[j].center)) * origin_list[j].points_size
                    center_sum += origin_list[j].center * origin_list[j].points_size
                    total_points += origin_list[j].points_size

                merge_center = center_sum / total_points
                merge_cov = P_PT_sum / total_points - np.outer(merge_center, merge_center)

                merge_plane.covariance = merge_cov
                merge_plane.center = merge_center
                merge_plane.points_size = total_points
                merge_plane.sub_plane_num = len(indices)

                # 重新计算法向量和半径
                eigenvals, eigenvecs = LA.eigh(merge_cov)
                merge_plane.normal = eigenvecs[:, 0]
                merge_plane.min_eigen_value = eigenvals[0]
                merge_plane.radius = np.sqrt(eigenvals[2])
                merge_plane.d = -np.dot(merge_plane.normal, merge_plane.center)

                # 设置合并后的Plane ID
                merge_plane.id = root + 1  # 使用root+1作为新ID
                merge_plane_list.append(merge_plane)

                if self.print_debug_info:
                    print(f"[MergePlane] Merged {len(indices)} planes into cluster {root}, "
                          f"points_size: {total_points}, center: {merge_center}, normal: {merge_plane.normal}")

        if self.print_debug_info:
            print(f"[MergePlane] End - merge_plane_list size: {len(merge_plane_list)}")

        return merge_plane_list

    def _binary_extractor(self, proj_plane_list, input_cloud):
        """
        对应 C++ BtcDescManager::binary_extractor
        P2-4: 增加大量调试日志
        """
        temp_binary_list = []
        last_normal = np.zeros(3)
        useful_proj_num = 0

        if self.print_debug_info:
            print(f"[BinaryExtractor] Start - proj_plane_list size: {len(proj_plane_list)}")

        for i in range(len(proj_plane_list)):
            prepare_binary_list = []
            proj_center = proj_plane_list[i].center
            proj_normal = proj_plane_list[i].normal.copy()
            plane_id = proj_plane_list[i].id  # P2-1: 使用Plane的ID
            if proj_normal[2] < 0:
                proj_normal = -proj_normal
            if LA.norm(proj_normal - last_normal) < 0.3 or LA.norm(proj_normal + last_normal) > 0.3:
                last_normal = proj_normal.copy()
                if self.print_debug_info:
                    print(f"[BinaryExtractor] Plane {i} - plane_id: {plane_id}, normal: {proj_normal}, center: {proj_center}")
                useful_proj_num += 1
                # P0-1: 传入plane_id
                self._extract_binary(proj_center, proj_normal, input_cloud, prepare_binary_list, plane_id)
                temp_binary_list.extend(prepare_binary_list)
                if self.print_debug_info:
                    print(f"[BinaryExtractor] After extract_binary - prepare_binary_list size: {len(prepare_binary_list)}")
                if useful_proj_num == self.config.proj_plane_num:
                    break

        if self.print_debug_info:
            print(f"[BinaryExtractor] Before NMS - temp_binary_list size: {len(temp_binary_list)}")

        self._non_maxi_suppression(temp_binary_list)

        if self.print_debug_info:
            print(f"[BinaryExtractor] After NMS - temp_binary_list size: {len(temp_binary_list)}")

        if self.config.useful_corner_num > len(temp_binary_list):
            return temp_binary_list
        else:
            temp_binary_list.sort(key=lambda b: b.summary, reverse=True)
            if self.print_debug_info:
                print(f"[BinaryExtractor] Select top {self.config.useful_corner_num} binaries")
            return temp_binary_list[:self.config.useful_corner_num]

    def _extract_binary(self, project_center, project_normal, input_cloud, binary_list, plane_id=-1):
        """
        对应 C++ BtcDescManager::extract_binary
        P0-2: 使用Gram-Schmidt方法生成稳定的局部坐标系
        P2-4: 增加大量调试日志
        """
        binary_list.clear()
        binary_min_dis = self.config.summary_min_thre
        resolution = self.config.proj_image_resolution
        dis_threshold_min = self.config.proj_dis_min
        dis_threshold_max = self.config.proj_dis_max
        high_inc = self.config.proj_image_high_inc
        line_filter_enable = self.config.line_filter_enable

        if self.print_debug_info:
            print(f"[ExtractBinary] Start - plane_id: {plane_id}, center: {project_center}, normal: {project_normal}")
            print(f"[ExtractBinary] Params - resolution: {resolution}, dis_threshold: [{dis_threshold_min}, {dis_threshold_max}], high_inc: {high_inc}")

        A = project_normal[0]
        B = project_normal[1]
        C = project_normal[2]
        D = -(A * project_center[0] + B * project_center[1] + C * project_center[2])

        # P0-2: 使用Gram-Schmidt方法生成稳定的局部坐标系，避免数值退化
        if abs(project_normal[2]) < 0.9:
            ref = np.array([0, 0, 1])  # 当法向不接近Z轴时，用Z轴作为参考
        else:
            ref = np.array([1, 0, 0])  # 当法向接近Z轴时，用X轴作为参考

        x_axis = np.cross(project_normal, ref)
        x_axis = x_axis / LA.norm(x_axis)
        y_axis = np.cross(project_normal, x_axis)
        y_axis = y_axis / LA.norm(y_axis)

        if self.print_debug_info:
            print(f"[ExtractBinary] Coordinate system - x_axis: {x_axis}, y_axis: {y_axis}")

        ax, bx, cx = x_axis
        dx = -(ax * project_center[0] + bx * project_center[1] + cx * project_center[2])
        ay, by, cy = y_axis
        dy = -(ay * project_center[0] + by * project_center[1] + cy * project_center[2])

        point_list_2d = []
        dis_list_2d = []
        point_list_3d = []

        n_sq = A * A + B * B + C * C

        for i in range(len(input_cloud)):
            x, y, z = input_cloud[i]
            dis = x * A + y * B + z * C + D
            if dis < dis_threshold_min or dis > dis_threshold_max:
                continue

            # 投影到平面
            cur_project = np.array([
                (-A * (B * y + C * z + D) + x * (B * B + C * C)) / n_sq,
                (-B * (A * x + C * z + D) + y * (A * A + C * C)) / n_sq,
                (-C * (A * x + B * y + D) + z * (A * A + B * B)) / n_sq,
            ])

            project_x = np.dot(cur_project, y_axis) + dy
            project_y = np.dot(cur_project, x_axis) + dx
            point_list_2d.append(np.array([project_x, project_y]))
            dis_list_2d.append(dis)
            point_list_3d.append(input_cloud[i].copy())

        if len(point_list_2d) <= 5:
            return

        point_list_2d = np.array(point_list_2d)
        dis_list_2d = np.array(dis_list_2d)

        min_x = point_list_2d[:, 0].min()
        max_x = point_list_2d[:, 0].max()
        min_y = point_list_2d[:, 1].min()
        max_y = point_list_2d[:, 1].max()

        segmen_base_num = 5
        segmen_len = segmen_base_num * resolution
        x_segment_num = int((max_x - min_x) / segmen_len) + 1
        y_segment_num = int((max_y - min_y) / segmen_len) + 1
        x_axis_len = int((max_x - min_x) / resolution + segmen_base_num)
        y_axis_len = int((max_y - min_y) / resolution + segmen_base_num)

        # 初始化 2D 数组
        img_count = np.zeros((x_axis_len, y_axis_len))
        mean_x = np.zeros((x_axis_len, y_axis_len))
        mean_y = np.zeros((x_axis_len, y_axis_len))
        dis_array = np.zeros((x_axis_len, y_axis_len))
        binary_container = [[None] * y_axis_len for _ in range(x_axis_len)]
        dis_container = [[[] for _ in range(y_axis_len)] for _ in range(x_axis_len)]

        for i in range(len(point_list_2d)):
            x_index = int((point_list_2d[i, 0] - min_x) / resolution)
            y_index = int((point_list_2d[i, 1] - min_y) / resolution)
            x_index = min(x_index, x_axis_len - 1)
            y_index = min(y_index, y_axis_len - 1)
            mean_x[x_index, y_index] += point_list_2d[i, 0]
            mean_y[x_index, y_index] += point_list_2d[i, 1]
            img_count[x_index, y_index] += 1
            dis_container[x_index][y_index].append(dis_list_2d[i])

        cut_num = int((dis_threshold_max - dis_threshold_min) / high_inc)

        for x in range(x_axis_len):
            for y in range(y_axis_len):
                if img_count[x, y] > 0:
                    occup_list = [False] * cut_num
                    cnt_list = [0] * cut_num
                    for d in dis_container[x][y]:
                        cnt_index = int((d - dis_threshold_min) / high_inc)
                        cnt_index = min(cnt_index, cut_num - 1)
                        cnt_list[cnt_index] += 1
                    segmnt_dis = 0
                    for ci in range(cut_num):
                        if cnt_list[ci] >= 1:
                            segmnt_dis += 1
                            occup_list[ci] = True
                    dis_array[x, y] = segmnt_dis

                    bdesc = BinaryDescriptor()
                    bdesc.occupy_array = occup_list
                    bdesc.summary = int(segmnt_dis)
                    # P0-1: 保存投影平面的法向量
                    bdesc.normal = project_normal.copy()
                    # P2-1: 保存投影平面的ID
                    bdesc.plane_id = plane_id
                    binary_container[x][y] = bdesc

        # 在 segment 中找最大 dis
        max_dis_list = []
        max_dis_x_index_list = []
        max_dis_y_index_list = []

        for x_seg in range(x_segment_num):
            for y_seg in range(y_segment_num):
                max_dis = 0
                max_x_idx = -10
                max_y_idx = -10
                for xi in range(x_seg * segmen_base_num,
                               min((x_seg + 1) * segmen_base_num, x_axis_len)):
                    for yi in range(y_seg * segmen_base_num,
                                   min((y_seg + 1) * segmen_base_num, y_axis_len)):
                        if dis_array[xi, yi] > max_dis:
                            max_dis = dis_array[xi, yi]
                            max_x_idx = xi
                            max_y_idx = yi
                if max_dis >= binary_min_dis:
                    max_dis_list.append(max_dis)
                    max_dis_x_index_list.append(max_x_idx)
                    max_dis_y_index_list.append(max_y_idx)

        direction_list = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for i in range(len(max_dis_list)):
            px = max_dis_x_index_list[i]
            py = max_dis_y_index_list[i]
            if px <= 0 or px >= x_axis_len - 1 or py <= 0 or py >= y_axis_len - 1:
                continue
            is_add = True

            if line_filter_enable:
                for dx_dir, dy_dir in direction_list:
                    p1x, p1y = px + dx_dir, py + dy_dir
                    p2x, p2y = px - dx_dir, py - dy_dir
                    if p1x <= 0 or p1x >= x_axis_len - 1 or p1y <= 0 or p1y >= y_axis_len - 1:
                        continue
                    if p2x <= 0 or p2x >= x_axis_len - 1 or p2y <= 0 or p2y >= y_axis_len - 1:
                        continue
                    threshold = dis_array[px, py] - 3
                    if dis_array[p1x, p1y] >= threshold:
                        if dis_array[p2x, p2y] >= 0.5 * dis_array[px, py]:
                            is_add = False
                    if dis_array[p2x, p2y] >= threshold:
                        if dis_array[p1x, p1y] >= 0.5 * dis_array[px, py]:
                            is_add = False
                    if dis_array[p1x, p1y] >= threshold:
                        if dis_array[p2x, p2y] >= threshold:
                            is_add = False
                    if dis_array[p2x, p2y] >= threshold:
                        if dis_array[p1x, p1y] >= threshold:
                            is_add = False

            if is_add:
                if img_count[px, py] > 0:
                    mean_px = mean_x[px, py] / img_count[px, py]
                    mean_py = mean_y[px, py] / img_count[px, py]
                    coord = mean_py * x_axis + mean_px * y_axis + project_center
                    bdesc = binary_container[px][py]
                    if bdesc is not None:
                        bdesc.location = coord.copy()
                        binary_list.append(bdesc)

    def _non_maxi_suppression(self, binary_list):
        """
        对应 C++ BtcDescManager::non_maxi_suppression
        P1-4: 增加score margin，避免大量descriptor被误删
        P2-4: 增加调试日志
        """
        if len(binary_list) <= 1:
            return

        if self.print_debug_info:
            print(f"[NMS] Start - binary_list size: {len(binary_list)}")

        locations = np.array([b.location for b in binary_list])
        summaries = [b.summary for b in binary_list]
        is_add = [True] * len(binary_list)
        radius = self.config.non_max_suppression_radius
        score_margin = self.config.nms_score_margin  # P1-4: 使用score margin

        if self.print_debug_info:
            print(f"[NMS] Parameters - radius: {radius}, score_margin: {score_margin}")

        if HAS_SCIPY:
            tree = ScipyKDTree(locations)
            for i in range(len(locations)):
                if not is_add[i]:
                    continue
                indices = tree.query_ball_point(locations[i], radius)
                for j in indices:
                    if j == i:
                        continue
                    # P1-4: 只有邻居summary > 自己summary + margin时才删除
                    if summaries[i] + score_margin <= summaries[j]:
                        is_add[i] = False
                        break
        else:
            # 暴力搜索
            for i in range(len(locations)):
                if not is_add[i]:
                    continue
                for j in range(len(locations)):
                    if i == j:
                        continue
                    if LA.norm(locations[i] - locations[j]) < radius:
                        # P1-4: 只有邻居summary > 自己summary + margin时才删除
                        if summaries[i] + score_margin <= summaries[j]:
                            is_add[i] = False
                            break

        filtered = [binary_list[i] for i in range(len(binary_list)) if is_add[i]]
        binary_list.clear()
        binary_list.extend(filtered)

        if self.print_debug_info:
            deleted_count = len(binary_list) - len(filtered)
            print(f"[NMS] End - kept: {len(filtered)}, deleted: {deleted_count}")

    def _generate_btc(self, binary_list, frame_id):
        """对应 C++ BtcDescManager::generate_btc"""
        scale = 1.0 / self.config.std_side_resolution
        btc_list = []
        feat_map = {}

        if len(binary_list) < 3:
            return btc_list

        locations = np.array([b.location for b in binary_list])

        if not HAS_SCIPY:
            # 简化: 暴力搜索 K 近邻
            return self._generate_btc_brute(binary_list, frame_id, scale, locations)

        tree = ScipyKDTree(locations)
        K = min(self.config.descriptor_near_num, len(binary_list))

        for i in range(len(binary_list)):
            dists, indices = tree.query(locations[i], k=K)
            if not hasattr(indices, '__len__'):
                indices = [indices]
                dists = [dists]

            for m in range(1, min(K - 1, len(indices) - 1)):
                for n in range(m + 1, min(K, len(indices))):
                    idx_m = indices[m]
                    idx_n = indices[n]

                    p1 = locations[i]
                    p2 = locations[idx_m]
                    p3 = locations[idx_n]

                    a = LA.norm(p1 - p2)
                    b = LA.norm(p1 - p3)
                    c = LA.norm(p3 - p2)

                    if (a > self.config.descriptor_max_len or
                            b > self.config.descriptor_max_len or
                            c > self.config.descriptor_max_len or
                            a < self.config.descriptor_min_len or
                            b < self.config.descriptor_min_len or
                            c < self.config.descriptor_min_len):
                        continue

                    # 排序 a <= b <= c (对应 C++ 的三重排序)
                    # l1, l2, l3 编码排序后的顶点来源
                    # C++ 中 l1=(1,2,0), l2=(1,0,3), l3=(0,2,3)
                    # l1 -> side a (between p1-p2), l2 -> side b (between p1-p3), l3 -> side c (between p2-p3)
                    sides = [(a, 1, 0), (b, 1, 1), (c, 0, 2)]
                    # 每项: (length, vertex_set_1, vertex_set_2)
                    # 实际上 C++ 中直接对 a,b,c 排序并交换 l1,l2,l3

                    # 对应 C++ 排序逻辑
                    la, lb, lc = a, b, c
                    # 记录顶点映射: l1->a边(1,2), l2->b边(1,0), l3->c边(0,2)
                    l1 = np.array([1, 2, 0])
                    l2 = np.array([1, 0, 3])
                    l3 = np.array([0, 2, 3])

                    # if a > b: swap a,b; swap l1,l2
                    if la > lb:
                        la, lb = lb, la
                        l1, l2 = l2.copy(), l1.copy()
                    # if b > c: swap b,c; swap l2,l3
                    if lb > lc:
                        lb, lc = lc, lb
                        l2, l3 = l3.copy(), l2.copy()
                    # if a > b: swap a,b; swap l1,l2
                    if la > lb:
                        la, lb = lb, la
                        l1, l2 = l2.copy(), l1.copy()

                    if abs(lc - (la + lb)) < 0.2:
                        continue

                    # BTC_LOC 量化
                    d_px = int(la * 1000)
                    d_py = int(lb * 1000)
                    d_pz = int(lc * 1000)
                    position = (d_px, d_py, d_pz)
                    if position in feat_map:
                        continue

                    # 根据 l1,l2,l3 确定三个顶点 A,B,C 和描述子
                    # l1 表示 a 边的两个端点集合, l2 表示 b 边, l3 表示 c 边
                    # A 是 a 边和 b 边的公共端点
                    # B 是 a 边和 c 边的公共端点
                    # C 是 b 边和 c 边的公共端点

                    points = [p1, p2, p3]
                    descs = [binary_list[i], binary_list[idx_m], binary_list[idx_n]]

                    # 找公共端点
                    def find_common_vertex(s1, s2):
                        # s1, s2 各有两个非零值，公共元素就是共享的端点索引
                        common = set()
                        for vi in range(3):
                            if s1[vi] != 0 and s2[vi] != 0:
                                common.add(vi)
                        # 但 C++ 逻辑不同：l1=[1,2,0] 表示 a边连接 vertex_1(p1) 和 vertex_2(p2)
                        # l2=[1,0,3] 表示 b边连接 vertex_1(p1) 和 vertex_0（这里0不代表实际顶点）
                        # 实际 C++ 中 l1,l2,l3 的含义是编码哪个顶点属于哪条边
                        # 重新理解: l1[0]==l2[0] -> vertex idx 0 (即 i)
                        #           l1[1]==l3[0] -> vertex idx m
                        #           l2[1]==l3[2] -> vertex idx n
                        # 不对，让我重新分析 C++ 代码
                        pass

                    # 直接按照 C++ 逻辑复现
                    # C++ 中:
                    # if (l1[0] == l2[0]): A = p1, binary_A = binary_list[i]
                    # elif (l1[1] == l2[1]): A = p2, binary_A = binary_list[idx_m]
                    # else: A = p3, binary_A = binary_list[idx_n]
                    #
                    # if (l1[0] == l3[0]): B = p1, binary_B = binary_list[i]
                    # elif (l1[1] == l3[1]): B = p2, binary_B = binary_list[idx_m]
                    # else: B = p3, binary_B = binary_list[idx_n]
                    #
                    # if (l2[0] == l3[0]): C = p1, binary_C = binary_list[i]
                    # elif (l2[1] == l3[1]): C = p2, binary_C = binary_list[idx_m]
                    # else: C = p3, binary_C = binary_list[idx_n]

                    def get_vertex_and_desc(l_a, l_b):
                        if l_a[0] == l_b[0]:
                            return p1.copy(), binary_list[i]
                        elif l_a[1] == l_b[1]:
                            return p2.copy(), binary_list[idx_m]
                        else:
                            return p3.copy(), binary_list[idx_n]

                    A_pt, binary_A = get_vertex_and_desc(l1, l2)
                    B_pt, binary_B = get_vertex_and_desc(l1, l3)
                    C_pt, binary_C = get_vertex_and_desc(l2, l3)

                    single_descriptor = BTCDesc()
                    single_descriptor.binary_A = binary_A
                    single_descriptor.binary_B = binary_B
                    single_descriptor.binary_C = binary_C
                    single_descriptor.center = (A_pt + B_pt + C_pt) / 3.0
                    single_descriptor.triangle = np.array([scale * la, scale * lb, scale * lc])
                    # C++ 中 angle 是法向量点积, 这里初始化为 0 (C++ 中 normal_1/2/3 未初始化)
                    single_descriptor.angle = np.zeros(3)
                    single_descriptor.frame_number = frame_id

                    feat_map[position] = True
                    btc_list.append(single_descriptor)

        return btc_list

    def _generate_btc_brute(self, binary_list, frame_id, scale, locations):
        """暴力搜索版 _generate_btc (无 scipy 时使用)"""
        btc_list = []
        feat_map = {}
        K = min(self.config.descriptor_near_num, len(binary_list))

        for i in range(len(binary_list)):
            dists = LA.norm(locations - locations[i], axis=1)
            sorted_indices = np.argsort(dists)
            if len(sorted_indices) < K:
                continue

            for m_idx in range(1, K - 1):
                for n_idx in range(m_idx + 1, K):
                    m = sorted_indices[m_idx]
                    n = sorted_indices[n_idx]

                    p1 = locations[i]
                    p2 = locations[m]
                    p3 = locations[n]

                    a = LA.norm(p1 - p2)
                    b = LA.norm(p1 - p3)
                    c = LA.norm(p3 - p2)

                    if (a > self.config.descriptor_max_len or
                            b > self.config.descriptor_max_len or
                            c > self.config.descriptor_max_len or
                            a < self.config.descriptor_min_len or
                            b < self.config.descriptor_min_len or
                            c < self.config.descriptor_min_len):
                        continue

                    la, lb, lc = a, b, c
                    l1 = np.array([1, 2, 0])
                    l2 = np.array([1, 0, 3])
                    l3 = np.array([0, 2, 3])

                    if la > lb:
                        la, lb = lb, la
                        l1, l2 = l2.copy(), l1.copy()
                    if lb > lc:
                        lb, lc = lc, lb
                        l2, l3 = l3.copy(), l2.copy()
                    if la > lb:
                        la, lb = lb, la
                        l1, l2 = l2.copy(), l1.copy()

                    if abs(lc - (la + lb)) < 0.2:
                        continue

                    d_px = int(la * 1000)
                    d_py = int(lb * 1000)
                    d_pz = int(lc * 1000)
                    position = (d_px, d_py, d_pz)
                    if position in feat_map:
                        continue

                    def get_vd(l_a, l_b):
                        if l_a[0] == l_b[0]:
                            return p1.copy(), binary_list[i]
                        elif l_a[1] == l_b[1]:
                            return p2.copy(), binary_list[m]
                        else:
                            return p3.copy(), binary_list[n]

                    A_pt, binary_A = get_vd(l1, l2)
                    B_pt, binary_B = get_vd(l1, l3)
                    C_pt, binary_C = get_vd(l2, l3)

                    single_descriptor = BTCDesc()
                    single_descriptor.binary_A = binary_A
                    single_descriptor.binary_B = binary_B
                    single_descriptor.binary_C = binary_C
                    single_descriptor.center = (A_pt + B_pt + C_pt) / 3.0
                    single_descriptor.triangle = np.array([scale * la, scale * lb, scale * lc])
                    single_descriptor.angle = np.zeros(3)
                    single_descriptor.frame_number = frame_id
                    feat_map[position] = True
                    btc_list.append(single_descriptor)

        return btc_list

    def _candidate_selector(self, current_STD_list):
        """对应 C++ BtcDescManager::candidate_selector"""
        current_frame_id = current_STD_list[0].frame_number
        candidate_matcher_vec = []

        max_dis = 50.0
        match_array = defaultdict(float)

        # 27 邻域
        voxel_round = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    voxel_round.append((x, y, z))

        useful_match = [False] * len(current_STD_list)
        useful_match_index = [[] for _ in range(len(current_STD_list))]
        useful_match_position = [[] for _ in range(len(current_STD_list))]

        for i, descriptor in enumerate(current_STD_list):
            dis_threshold = LA.norm(descriptor.triangle) * self.config.rough_dis_threshold

            # P1-1: 使用可配置分辨率进行量化
            resolution = self.config.std_side_resolution

            for voxel_inc in voxel_round:
                px = int(descriptor.triangle[0] / resolution + voxel_inc[0])
                py = int(descriptor.triangle[1] / resolution + voxel_inc[1])
                pz = int(descriptor.triangle[2] / resolution + voxel_inc[2])
                position = (px, py, pz)

                # voxel_center调整：考虑分辨率
                voxel_center = np.array([px + 0.5, py + 0.5, pz + 0.5]) * resolution
                # 调整距离阈值，使用分辨率相关的值（1.5倍分辨率）
                if LA.norm(descriptor.triangle - voxel_center) < 1.5 * resolution:
                    if position in self.data_base:
                        for j, db_btc in enumerate(self.data_base[position]):
                            if (descriptor.frame_number - db_btc.frame_number) > self.config.skip_near_num:
                                dis = LA.norm(descriptor.triangle - db_btc.triangle)
                                if dis < dis_threshold:
                                    similarity = (
                                        binary_similarity(descriptor.binary_A, db_btc.binary_A) +
                                        binary_similarity(descriptor.binary_B, db_btc.binary_B) +
                                        binary_similarity(descriptor.binary_C, db_btc.binary_C)
                                    ) / 3.0
                                    if similarity > self.config.similarity_threshold:
                                        useful_match[i] = True
                                        useful_match_position[i].append(position)
                                        useful_match_index[i].append(j)

        # 统计投票
        index_recorder = []   # (query_idx, db_idx_in_position)
        match_list_index = [] # frame_number

        for i in range(len(current_STD_list)):
            if useful_match[i]:
                for j in range(len(useful_match_index[i])):
                    db_btc = self.data_base[useful_match_position[i][j]][useful_match_index[i][j]]
                    match_array[db_btc.frame_number] += 1
                    index_recorder.append((i, j))
                    match_list_index.append(db_btc.frame_number)

        # 选择 top 候选
        for cnt in range(self.config.candidate_num):
            max_vote = 1
            max_vote_index = -1
            for frame_id, vote in match_array.items():
                if vote > max_vote:
                    max_vote = vote
                    max_vote_index = frame_id
            if max_vote_index >= 0 and max_vote >= 5:
                match_array[max_vote_index] = 0
                match_triangle_list = BTCMatchList()
                match_triangle_list.match_frame = max_vote_index
                match_triangle_list.match_id = (current_frame_id, max_vote_index)

                for i in range(len(index_recorder)):
                    if match_list_index[i] == max_vote_index:
                        q_idx, db_j = index_recorder[i]
                        single_match_pair = (
                            current_STD_list[q_idx],
                            self.data_base[useful_match_position[q_idx][db_j]][useful_match_index[q_idx][db_j]]
                        )
                        match_triangle_list.match_list.append(single_match_pair)

                candidate_matcher_vec.append(match_triangle_list)
            else:
                break

        return candidate_matcher_vec

    def _candidate_verify(self, candidate_matcher):
        """对应 C++ BtcDescManager::candidate_verify"""
        sucess_match_list = []
        dis_threshold = 3.0

        skip_len = max(1, int(len(candidate_matcher.match_list) / 50))
        use_size = len(candidate_matcher.match_list) // skip_len

        if use_size == 0:
            return -1.0, (np.zeros(3), np.eye(3)), []

        vote_list = []

        for i in range(use_size):
            single_pair = candidate_matcher.match_list[i * skip_len]
            test_rot, test_t = self._triangle_solver(single_pair)
            vote = 0
            for j in range(len(candidate_matcher.match_list)):
                verify_pair = candidate_matcher.match_list[j]
                A_transform = test_rot @ verify_pair[0].binary_A.location + test_t
                B_transform = test_rot @ verify_pair[0].binary_B.location + test_t
                C_transform = test_rot @ verify_pair[0].binary_C.location + test_t
                dis_A = LA.norm(A_transform - verify_pair[1].binary_A.location)
                dis_B = LA.norm(B_transform - verify_pair[1].binary_B.location)
                dis_C = LA.norm(C_transform - verify_pair[1].binary_C.location)
                if dis_A < dis_threshold and dis_B < dis_threshold and dis_C < dis_threshold:
                    vote += 1
            vote_list.append(vote)

        max_vote_index = 0
        max_vote = 0
        for i in range(len(vote_list)):
            if vote_list[i] > max_vote:
                max_vote_index = i
                max_vote = vote_list[i]

        if max_vote >= 4:
            best_pair = candidate_matcher.match_list[max_vote_index * skip_len]
            best_rot, best_t = self._triangle_solver(best_pair)
            relative_pose = (best_t, best_rot)

            for j in range(len(candidate_matcher.match_list)):
                verify_pair = candidate_matcher.match_list[j]
                A_transform = best_rot @ verify_pair[0].binary_A.location + best_t
                B_transform = best_rot @ verify_pair[0].binary_B.location + best_t
                C_transform = best_rot @ verify_pair[0].binary_C.location + best_t
                dis_A = LA.norm(A_transform - verify_pair[1].binary_A.location)
                dis_B = LA.norm(B_transform - verify_pair[1].binary_B.location)
                dis_C = LA.norm(C_transform - verify_pair[1].binary_C.location)
                if dis_A < dis_threshold and dis_B < dis_threshold and dis_C < dis_threshold:
                    sucess_match_list.append(verify_pair)

            verify_score = self._plane_geometric_verify(
                len(self.plane_cloud_vec) - 1,
                candidate_matcher.match_id[1],
                relative_pose
            )
            return verify_score, relative_pose, sucess_match_list
        else:
            return -1.0, (np.zeros(3), np.eye(3)), []

    def _triangle_solver(self, std_pair):
        """对应 C++ BtcDescManager::triangle_solver — SVD 求解旋转和平移"""
        first, second = std_pair
        src = np.column_stack([
            first.binary_A.location - first.center,
            first.binary_B.location - first.center,
            first.binary_C.location - first.center,
        ])
        ref = np.column_stack([
            second.binary_A.location - second.center,
            second.binary_B.location - second.center,
            second.binary_C.location - second.center,
        ])
        covariance = src @ ref.T
        U, S, Vt = LA.svd(covariance)
        V = Vt.T
        U_full = U
        rot = V @ U_full.T
        if LA.det(rot) < 0:
            K = np.diag([1, 1, -1])
            rot = V @ K @ U_full.T
        t = -rot @ first.center + second.center
        return rot, t

    def _plane_geometric_verify(self, source_idx, target_idx, transform):
        """对应 C++ BtcDescManager::plane_geometric_verify"""
        t, rot = transform
        if source_idx < 0 or source_idx >= len(self.plane_cloud_vec):
            return -1.0
        if target_idx < 0 or target_idx >= len(self.plane_cloud_vec):
            return -1.0

        source_cloud = self.plane_cloud_vec[source_idx]
        target_cloud = self.plane_cloud_vec[target_idx]

        if len(source_cloud) == 0 or len(target_cloud) == 0:
            return -1.0

        target_pts = np.array([c for c, n in target_cloud])
        target_normals = np.array([n for c, n in target_cloud])

        if not HAS_SCIPY:
            return self._plane_geometric_verify_brute(source_cloud, target_pts, target_normals, rot, t)

        kd_tree = ScipyKDTree(target_pts)
        useful_match = 0
        normal_threshold = self.config.normal_threshold
        dis_threshold = self.config.dis_threshold

        for center_s, normal_s in source_cloud:
            pi = rot @ center_s + t
            ni = rot @ normal_s
            _, idx = kd_tree.query(pi, k=1)
            tpi = target_pts[idx]
            tni = target_normals[idx]
            normal_inc = LA.norm(ni - tni)
            normal_add = LA.norm(ni + tni)
            point_to_plane = abs(np.dot(tni, pi - tpi))
            if (normal_inc < normal_threshold or normal_add < normal_threshold) and point_to_plane < dis_threshold:
                useful_match += 1

        return useful_match / len(source_cloud)

    def _plane_geometric_verify_brute(self, source_cloud, target_pts, target_normals, rot, t):
        """暴力版平面几何验证"""
        useful_match = 0
        normal_threshold = self.config.normal_threshold
        dis_threshold = self.config.dis_threshold

        for center_s, normal_s in source_cloud:
            pi = rot @ center_s + t
            ni = rot @ normal_s
            dists = LA.norm(target_pts - pi, axis=1)
            idx = np.argmin(dists)
            if dists[idx] > 5.0:
                continue
            tpi = target_pts[idx]
            tni = target_normals[idx]
            normal_inc = LA.norm(ni - tni)
            normal_add = LA.norm(ni + tni)
            point_to_plane = abs(np.dot(tni, pi - tpi))
            if (normal_inc < normal_threshold or normal_add < normal_threshold) and point_to_plane < dis_threshold:
                useful_match += 1

        return useful_match / max(len(source_cloud), 1)


# ======================== 辅助函数 ========================

def calc_triangle_dis(match_std_list):
    """对应 C++ calc_triangle_dis"""
    if len(match_std_list) == 0:
        return -1.0
    total = 0.0
    for first, second in match_std_list:
        total += LA.norm(first.triangle - second.triangle) / LA.norm(first.triangle)
    return total / len(match_std_list)


def calc_binary_similarity(match_std_list):
    """对应 C++ calc_binary_similaity"""
    if len(match_std_list) == 0:
        return -1.0
    total = 0.0
    for first, second in match_std_list:
        total += (
            binary_similarity(first.binary_A, second.binary_A) +
            binary_similarity(first.binary_B, second.binary_B) +
            binary_similarity(first.binary_C, second.binary_C)
        ) / 3.0
    return total / len(match_std_list)
