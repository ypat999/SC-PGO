"""
纯 numpy 建图工具 — 读取 PCD + 位姿变换 + 合并保存

依赖: numpy (零第三方库依赖)
"""
import os
import struct
import numpy as np
from numpy import linalg as LA

this_folder = os.path.dirname(os.path.abspath(__file__))

jet_table = np.load(this_folder + '/jet_table.npy')
bone_table = np.load(this_folder + '/bone_table.npy')

color_table = jet_table
color_table_len = color_table.shape[0]

data_dir = "/home/ywj/save_data/"  # should end with /
scan_dir = data_dir + "Scans"
scan_files = os.listdir(scan_dir)
scan_files.sort()

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

# 优化后的位姿文件可能只有关键帧，改用 odom_poses.txt（所有帧的原始位姿）
poses = []
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
    scan_xyz_local, scan_intensity = read_pcd_with_intensity(scan_path)
    scan_pose = poses[node_idx]

    # 变换到全局坐标系
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
