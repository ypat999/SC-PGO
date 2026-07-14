/**
 * @file dynamic_remove.hpp
 * @brief Dynamic point removal for SC-PGO map saving
 *
 * Provides Temporal method (multi-frame consistency) for dynamic point removal
 * and isolated point removal (outlier filtering).
 * No Raycast method is implemented.
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <tuple>

namespace DynamicRemove {

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;

struct Config {
    bool enable = true;
    bool isolated_removal = false;
    int method = 0;              // 0 = Temporal
    float grid_size = 0.2f;
    int min_neighbors = 2;
    int frame_window = 1;
    std::string output_dir;      // directory for intermediate per-frame filtered PCDs
};

struct VoxelKey {
    int x, y, z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        return std::hash<int>()(key.x) ^ (std::hash<int>()(key.y) << 1)
               ^ (std::hash<int>()(key.z) << 2);
    }
};

inline VoxelKey pointToVoxelKey(const PointType& point, float grid_size) {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(point.x / grid_size));
    key.y = static_cast<int>(std::floor(point.y / grid_size));
    key.z = static_cast<int>(std::floor(point.z / grid_size));
    return key;
}

/**
 * @brief Built occupancy grid from a point cloud (unordered_set of voxels).
 */
class OccupancyGrid {
public:
    explicit OccupancyGrid(float grid_size) : grid_size_(grid_size) {}

    void insertCloud(const CloudPtr& cloud) {
        for (const auto& pt : cloud->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
                continue;
            occupied_voxels_.insert(pointToVoxelKey(pt, grid_size_));
        }
    }

    bool isOccupied(const VoxelKey& key) const {
        return occupied_voxels_.find(key) != occupied_voxels_.end();
    }

    size_t size() const { return occupied_voxels_.size(); }

private:
    float grid_size_;
    std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels_;
};

/**
 * @brief Temporal dynamic point removal.
 *
 * For each frame, a point is considered dynamic if its voxel is NOT occupied
 * in any of the neighbouring frames within [frame_idx - window, frame_idx + window].
 *
 * @param frames        Input point clouds (one per keyframe).
 * @param config        Removal configuration.
 * @param out_filtered  Output: per-frame filtered point clouds (same count as input).
 * @param out_stats     Optional: (total_points, removed_points) statistics.
 */
void filterDynamicPointsTemporal(
    const std::vector<CloudPtr>& frames,
    const Config& config,
    std::vector<CloudPtr>& out_filtered,
    std::pair<int, int>* out_stats = nullptr);

/**
 * @brief Remove isolated (outlier) points from a single cloud.
 *
 * A point is considered isolated if fewer than min_neighbors of its 26-neighbour
 * voxels are occupied.
 *
 * @param cloud   Input point cloud.
 * @param config  Removal configuration (uses grid_size, min_neighbors).
 * @return        Filtered point cloud.
 */
CloudPtr removeIsolatedPoints(const CloudPtr& cloud, const Config& config);

/**
 * @brief Save per-frame filtered clouds to intermediate PCD files.
 *
 * @param filtered_frames  Per-frame filtered point clouds.
 * @param output_dir       Output directory.
 * @param filenames        Original filenames (basenames), used to generate "filtered_<name>".
 * @return                 Number of files saved.
 */
int saveFilteredFrames(
    const std::vector<CloudPtr>& filtered_frames,
    const std::string& output_dir,
    const std::vector<std::string>& filenames);

/**
 * @brief Merge all filtered_*.pcd files in a directory into a single cloud.
 *
 * @param input_dir  Directory containing filtered_*.pcd files.
 * @return           Merged point cloud (empty if no files found).
 */
CloudPtr mergeFilteredPCDs(const std::string& input_dir);

}  // namespace DynamicRemove
