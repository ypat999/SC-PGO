"""
纯 numpy 建图工具 — 读取 PCD + 位姿变换 + 合并保存

支持两种数据格式（自动识别）:
  - KITTI  : data_dir 下 odom_poses.txt(或 optimized_poses.txt) + Scans/000000.pcd
             pcd 为单帧 body 系，用位姿变换到世界系后合并
  - Super-LIO 原生: data_dir 下 scans_*.pcd + scans_*.txt 成对
             pcd 是原始 odom 的**世界系**累积点云，本身不含 PGO 结果；
             若目录内有 optimized_poses.txt，则做 PGO 矫正后合并:
                 p_body = T_raw^-1 @ p_raw  →  p_world = T_opt @ p_body
             无 optimized_poses.txt 时退化为直接使用原始世界系点云
             txt 格式: timestamp tx ty tz qx qy qz qw

依赖: numpy (零第三方库依赖)
"""
import os
import re
import struct
import numpy as np
from numpy import linalg as LA

this_folder = os.path.dirname(os.path.abspath(__file__))

jet_table = np.load(this_folder + '/jet_table.npy')
bone_table = np.load(this_folder + '/bone_table.npy')

color_table = jet_table
color_table_len = color_table.shape[0]

# ------------------------------------------------------------------
# 数据目录（改这里即可，目录需以 / 结尾）
#   KITTI     格式: 例如 "/home/ywj/save_data/"
#   Super-LIO 格式: 例如 "/home/ywj/slam_data/PCD/"
# ------------------------------------------------------------------
data_dir = "/home/ywj/slam_data/PCD/"   # should end with /
data_format = "auto"                    # "auto" | "kitti" | "superlio"
use_optimized_poses = True              # 是否用 optimized_poses.txt(SC-PGO结果) 做 PGO 矫正

if not data_dir.endswith('/'):
    data_dir += '/'


def _trailing_int(stem):
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else None


def _find_superlio_pairs(d):
    """查找 scans_*.pcd + scans_*.txt 成对文件，按数字索引排序"""
    pairs = []
    if not os.path.isdir(d):
        return pairs
    for fn in os.listdir(d):
        if not fn.endswith('.pcd'):
            continue
        txt = fn[:-4] + '.txt'
        if os.path.exists(os.path.join(d, txt)):
            pairs.append((_trailing_int(fn[:-4]), fn, txt))
    pairs.sort(key=lambda e: (e[0] is None, e[0] if e[0] is not None else 0, e[1]))
    return pairs


def _detect_format(d):
    if data_format != 'auto':
        return data_format
    if _find_superlio_pairs(d):
        return 'superlio'
    return 'kitti'


data_format_detected = _detect_format(data_dir)
body_frame = (data_format_detected == 'superlio')  # Super-LIO 的 pcd 已是世界系

if data_format_detected == 'kitti':
    scan_dir = data_dir + 'Scans' if os.path.isdir(data_dir + 'Scans') else data_dir
    scan_files = sorted(f for f in os.listdir(scan_dir) if f.endswith('.pcd'))
    _superlio_pose_files = None
else:
    scan_dir = data_dir
    _pairs = _find_superlio_pairs(data_dir)
    scan_files = [p[1] for p in _pairs]
    _superlio_pose_files = [p[2] for p in _pairs]

print(f"[Format] {data_format_detected} | 扫描目录: {scan_dir} | 帧数: {len(scan_files)}")

##########################
# User only consider this block
##########################


scan_idx_range_to_stack = [0, len(scan_files) - 1] # if you want a whole map, use [0, len(scan_files)]
node_skip = 1

num_points_in_a_scan = 10000  # 仅用作预估提示，实际使用动态扩容

is_near_removal = True
thres_near_removal = 2

##########################


def read_pcd_with_intensity(filepath):
    """读取 PCD，返回 (xyz, intensity) 两个 Nx3 和 Nx1 数组"""
    with open(filepath, 'rb') as f:
        header_lines = []
        fields, sizes, types = [], [], []
        points_count = 0
        data_type = 'ascii'
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                sizes = [int(s) for s in line.split()[1:]]
            elif line.startswith('TYPE'):
                types = line.split()[1:]
            elif line.startswith('POINTS'):
                points_count = int(line.split()[1])
            elif line.startswith('DATA'):
                data_type = line.split()[1].lower()
                break

    point_step = sum(sizes)
    header_size = sum(len(l) + 1 for l in header_lines)

    with open(filepath, 'rb') as f:
        f.seek(header_size)
        raw = f.read(points_count * point_step)

    # 字段偏移量
    offsets = []
    o = 0
    for s in sizes:
        offsets.append(o)
        o += s

    def field_idx(name):
        return fields.index(name) if name in fields else -1

    x_i = field_idx('x')
    y_i = field_idx('y')
    z_i = field_idx('z')
    int_i = field_idx('intensity')

    xyz = np.zeros((points_count, 3), dtype=np.float64)
    intensity = np.zeros((points_count, 1), dtype=np.float64)

    for i in range(points_count):
        base = i * point_step
        if x_i >= 0:
            x = struct.unpack_from('<f', raw, base + offsets[x_i])[0]
            y = struct.unpack_from('<f', raw, base + offsets[y_i])[0]
            z = struct.unpack_from('<f', raw, base + offsets[z_i])[0]
            xyz[i] = [x, y, z]
        if int_i >= 0:
            fmt = '<f' if types[int_i] == 'F' else '<I'
            intensity[i] = struct.unpack_from(fmt, raw, base + offsets[int_i])[0]

    return xyz, intensity


def save_pcd_with_intensity(filepath, xyz, intensity=None, ascii=False):
    """保存 PCD，含 intensity"""
    n = len(xyz)
    has_intensity = intensity is not None and len(intensity) == n

    with open(filepath, 'wb') as f:
        # header
        f.write(b'# .PCD v0.7 - Point Cloud Data file format\n')
        f.write(b'VERSION 0.7\n')
        if has_intensity:
            f.write(b'FIELDS x y z intensity\n')
            f.write(b'SIZE 4 4 4 4\n')
            f.write(b'TYPE F F F F\n')
            f.write(b'COUNT 1 1 1 1\n')
        else:
            f.write(b'FIELDS x y z\n')
            f.write(b'SIZE 4 4 4\n')
            f.write(b'TYPE F F F\n')
            f.write(b'COUNT 1 1 1\n')
        f.write(f'WIDTH {n}\n'.encode())
        f.write(b'HEIGHT 1\n')
        f.write(f'POINTS {n}\n'.encode())

        if ascii:
            f.write(b'DATA ascii\n')
            if has_intensity:
                for i in range(n):
                    f.write(f'{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {intensity[i,0]:.6f}\n'.encode())
            else:
                for i in range(n):
                    f.write(f'{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f}\n'.encode())
        else:
            f.write(b'DATA binary\n')
            buf = bytearray(n * (16 if has_intensity else 12))
            if has_intensity:
                for i in range(n):
                    off = i * 16
                    struct.pack_into('<ffff', buf, off,
                                     float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                                     float(intensity[i, 0]))
            else:
                for i in range(n):
                    off = i * 12
                    struct.pack_into('<fff', buf, off,
                                     float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
            f.write(buf)


# ====================== 主流程 ======================

print("Merging scans from", scan_idx_range_to_stack[0], "to", scan_idx_range_to_stack[1])


def _quat_xyzw_to_matrix(qx, qy, qz, qw):
    """四元数 (x,y,z,w) -> 3x3 旋转矩阵"""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def _load_superlio_pose(filepath):
    """Super-LIO txt 格式: timestamp tx ty tz qx qy qz qw -> 4x4 位姿"""
    with open(filepath, 'r') as f:
        for line in f:
            v = line.split()
            if len(v) < 8:
                continue
            T = np.eye(4)
            T[:3, 3] = [float(v[1]), float(v[2]), float(v[3])]
            T[:3, :3] = _quat_xyzw_to_matrix(float(v[4]), float(v[5]),
                                             float(v[6]), float(v[7]))
            return T
    return np.eye(4)


poses = []       # 原始位姿: superlio=scans_*.txt / kitti=odom_poses.txt
poses_opt = []   # PGO 优化位姿: optimized_poses.txt (可能为空)

if body_frame:
    # Super-LIO: 原始位姿取自每帧的 scans_*.txt
    poses = [_load_superlio_pose(os.path.join(scan_dir, pf))
             for pf in _superlio_pose_files]
    print(f"[Load] {len(poses)} 个原始位姿, 来源: scans_*.txt (Super-LIO 原生)")
else:
    # 优先使用 PGO 优化位姿；若不存在则回退到所有帧的原始位姿
    poses_file = data_dir + "optimized_poses.txt"
    if not os.path.exists(poses_file):
        poses_file = data_dir + "odom_poses.txt"
    with open(poses_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pose_SE3 = np.asarray([float(i) for i in line.split()])
            pose_SE3 = np.vstack((np.reshape(pose_SE3, (3, 4)), np.asarray([0, 0, 0, 1])))
            poses.append(pose_SE3)
    print(f"[Load] {len(poses)} 个位姿, 来源: {poses_file}")

# Super-LIO: pcd 已是原始 odom 世界系，需用 PGO 优化位姿做相对矫正
#   p_body = T_raw^-1 @ p_raw ; p_world_pgo = T_opt @ p_body
if body_frame and use_optimized_poses:
    opt_path = data_dir + "optimized_poses.txt"
    if os.path.exists(opt_path):
        _opt = []
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = np.asarray([float(i) for i in line.split()])
                _opt.append(np.vstack((np.reshape(v, (3, 4)), np.asarray([0, 0, 0, 1]))))
        if len(_opt) == len(scan_files):
            poses_opt = _opt
            print(f"[Load] {len(poses_opt)} 个优化位姿, 来源: {opt_path} -> 启用 PGO 矫正")
        else:
            print(f"[WARN] optimized_poses.txt 帧数({len(_opt)}) 与扫描帧数({len(scan_files)}) "
                  f"不一致，已忽略 PGO 矫正")
    else:
        print(f"[WARN] 未找到 {opt_path}，不做 PGO 矫正（使用原始 odom 位姿）")

assert scan_idx_range_to_stack[1] > scan_idx_range_to_stack[0]

# 使用 list 动态追加，避免预分配不足
xyz_list = []
intensity_list = []

for node_idx in range(len(scan_files)):
    if node_idx < scan_idx_range_to_stack[0] or node_idx >= scan_idx_range_to_stack[1]:
        continue

    if node_idx != scan_idx_range_to_stack[0] and ((node_idx - scan_idx_range_to_stack[0]) % node_skip) != 0:
        continue

    print("read keyframe scan idx", node_idx)

    scan_path = os.path.join(scan_dir, scan_files[node_idx])
    scan_xyz_raw, scan_intensity = read_pcd_with_intensity(scan_path)
    scan_pose = poses[node_idx]

    if body_frame:
        # Super-LIO: pcd 已是原始 odom 世界系，先反变换回 body 系
        R = scan_pose[:3, :3]
        t = scan_pose[:3, 3]
        scan_xyz_local = (R.T @ (scan_xyz_raw - t).T).T
        # PGO 矫正: 用优化位姿替代原始位姿正变换回世界系
        T_use = poses_opt[node_idx] if poses_opt else scan_pose
        scan_xyz_global = (T_use[:3, :3] @ scan_xyz_local.T).T + T_use[:3, 3]
    else:
        # KITTI: pcd 为单帧 body 系，用位姿变换到世界系
        scan_xyz_local = scan_xyz_raw
        scan_xyz_h = np.column_stack([scan_xyz_local, np.ones(len(scan_xyz_local))])
        scan_xyz_global = (scan_pose @ scan_xyz_h.T).T[:, :3]

    # 去除近处点
    if is_near_removal:
        scan_ranges = LA.norm(scan_xyz_local, axis=1)
        mask = scan_ranges > thres_near_removal
        scan_xyz_global = scan_xyz_global[mask]
        scan_intensity = scan_intensity[mask]

    xyz_list.append(scan_xyz_global)
    intensity_list.append(scan_intensity.flatten())
    curr_count = sum(len(x) for x in xyz_list)
    print(curr_count)

np_xyz_all = np.vstack(xyz_list)
np_intensity_all = np.hstack(intensity_list)[:, np.newaxis]

map_name = data_dir + "map_" + str(scan_idx_range_to_stack[0]) + "_to_" + str(scan_idx_range_to_stack[1]) + "_with_intensity.pcd"
save_pcd_with_intensity(map_name, np_xyz_all, np_intensity_all)
print("intensity map saved (path:", map_name, ")")
