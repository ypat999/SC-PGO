/**
 * @file dynamic_remove.cpp
 * @brief Dynamic point removal implementation (Temporal + Isolated only)
 */

#include "dynamic_remove.hpp"

#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

namespace DynamicRemove {

// ---------------------------------------------------------------------------
// Temporal dynamic point removal
// ---------------------------------------------------------------------------
void filterDynamicPointsTemporal(
    const std::vector<CloudPtr>& frames,
    const Config& config,
    std::vector<CloudPtr>& out_filtered,
    std::pair<int, int>* out_stats)
{
    out_filtered.clear();
    if (frames.empty()) return;

    size_t n = frames.size();

    // Build occupancy grid for every frame
    std::vector<OccupancyGrid> grids;
    grids.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        grids.emplace_back(config.grid_size);
        grids[i].insertCloud(frames[i]);
    }

    out_filtered.resize(n);
    int total = 0, removed = 0;

    for (size_t i = 0; i < n; ++i) {
        out_filtered[i].reset(new PointCloudType());
        const auto& frame = frames[i];
        int frame_removed = 0;

        for (const auto& pt : frame->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }
            ++total;
            VoxelKey key = pointToVoxelKey(pt, config.grid_size);

            bool is_dynamic = true;
            int w = config.frame_window;

            for (int off = -w; off <= w; ++off) {
                if (off == 0) continue;
                int ni = static_cast<int>(i) + off;
                if (ni < 0 || ni >= static_cast<int>(n)) continue;

                if (grids[ni].isOccupied(key)) {
                    is_dynamic = false;
                    break;
                }
            }

            if (is_dynamic) {
                ++removed;
                ++frame_removed;
            } else {
                out_filtered[i]->points.push_back(pt);
            }
        }

        out_filtered[i]->width = out_filtered[i]->points.size();
        out_filtered[i]->height = 1;
        out_filtered[i]->is_dense = true;
    }

    if (out_stats) {
        *out_stats = {total, removed};
    }
}

// ---------------------------------------------------------------------------
// Isolated (outlier) point removal
// ---------------------------------------------------------------------------
CloudPtr removeIsolatedPoints(const CloudPtr& cloud, const Config& config)
{
    if (!config.isolated_removal || cloud->empty()) {
        return cloud;
    }

    OccupancyGrid grid(config.grid_size);
    grid.insertCloud(cloud);

    CloudPtr filtered(new PointCloudType());
    int total = static_cast<int>(cloud->size());
    int isolated = 0;

    // 26-neighbour offsets
    static const std::vector<std::tuple<int, int, int>> offsets = {
        {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
        {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
        {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
        { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
        { 0, 0,-1},              { 0, 0, 1},
        { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
        { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
        { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
        { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}
    };

    for (const auto& pt : cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            continue;
        }
        VoxelKey key = pointToVoxelKey(pt, config.grid_size);

        int neighbour_count = 0;
        for (const auto& [dx, dy, dz] : offsets) {
            VoxelKey nk{key.x + dx, key.y + dy, key.z + dz};
            if (grid.isOccupied(nk)) {
                ++neighbour_count;
            }
        }

        if (neighbour_count < config.min_neighbors) {
            ++isolated;
        } else {
            filtered->points.push_back(pt);
        }
    }

    filtered->width = filtered->points.size();
    filtered->height = 1;
    filtered->is_dense = true;

    std::cout << "[DynamicRemove] Isolated removal: " << isolated << " / "
              << total << " points removed ("
              << (100.0 * isolated / total) << "%)" << std::endl;

    return filtered;
}

// ---------------------------------------------------------------------------
// Save per-frame filtered clouds to intermediate PCD files
// ---------------------------------------------------------------------------
int saveFilteredFrames(
    const std::vector<CloudPtr>& filtered_frames,
    const std::string& output_dir,
    const std::vector<std::string>& filenames)
{
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    int saved = 0;
    for (size_t i = 0; i < filtered_frames.size(); ++i) {
        const auto& cloud = filtered_frames[i];
        if (cloud->empty()) continue;

        std::string basename;
        if (i < filenames.size()) {
            basename = filenames[i];
        } else {
            basename = std::to_string(i) + ".pcd";
        }

        std::string out_path = output_dir + "/filtered_" + basename;
        if (pcl::io::savePCDFileBinary(out_path, *cloud) == 0) {
            ++saved;
        } else {
            std::cerr << "[DynamicRemove] Failed to save: " << out_path << std::endl;
        }
    }

    std::cout << "[DynamicRemove] Saved " << saved << " filtered intermediate PCDs to "
              << output_dir << std::endl;
    return saved;
}

// ---------------------------------------------------------------------------
// Merge all filtered_*.pcd files into a single cloud
// ---------------------------------------------------------------------------
CloudPtr mergeFilteredPCDs(const std::string& input_dir)
{
    CloudPtr merged(new PointCloudType());

    if (!fs::exists(input_dir)) {
        std::cerr << "[DynamicRemove] Input directory does not exist: " << input_dir << std::endl;
        return merged;
    }

    std::vector<std::string> filtered_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        std::string name = entry.path().filename().string();
        if (name.find("filtered_") == 0 && entry.path().extension() == ".pcd") {
            filtered_files.push_back(entry.path().string());
        }
    }
    std::sort(filtered_files.begin(), filtered_files.end());

    int merged_count = 0;
    for (const auto& f : filtered_files) {
        CloudPtr cloud(new PointCloudType());
        if (pcl::io::loadPCDFile<PointType>(f, *cloud) == 0) {
            *merged += *cloud;
            ++merged_count;
        }
    }

    std::cout << "[DynamicRemove] Merged " << merged_count << " filtered PCDs -> "
              << merged->size() << " total points" << std::endl;
    return merged;
}

}  // namespace DynamicRemove
