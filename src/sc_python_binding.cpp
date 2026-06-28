/**
 * ScanContext Python Binding - 通过pybind11将C++ SCManager暴露给Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "scancontext/Scancontext.h"

namespace py = pybind11;

using SCPointType = pcl::PointXYZI;
using PointType = pcl::PointXYZI;

// Helper function: numpy array to pcl PointCloud
pcl::PointCloud<PointType>::Ptr numpy_to_pcl(py::array_t<float> array) {
    auto buf = array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input array must be Nx3");
    }

    float *ptr = (float *) buf.ptr;
    size_t num_points = buf.shape[0];

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    cloud->reserve(num_points);
    for (size_t i = 0; i < num_points; i++) {
        PointType point;
        point.x = ptr[i * 3 + 0];
        point.y = ptr[i * 3 + 1];
        point.z = ptr[i * 3 + 2];
        point.intensity = 1.0;  // default intensity
        cloud->push_back(point);
    }
    return cloud;
}

// Helper function: pcl PointCloud to numpy array
py::array_t<float> pcl_to_numpy(pcl::PointCloud<PointType>::Ptr cloud) {
    size_t num_points = cloud->size();
    py::array_t<float> array(std::vector<size_t>{num_points, 3});
    auto buf = array.request();
    float *ptr = (float *) buf.ptr;

    for (size_t i = 0; i < num_points; i++) {
        ptr[i * 3 + 0] = cloud->points[i].x;
        ptr[i * 3 + 1] = cloud->points[i].y;
        ptr[i * 3 + 2] = cloud->points[i].z;
    }
    return array;
}

/**
 * Python wrapper for SCManager
 */
class PySCManager {
private:
    SCManager manager_;

public:
    PySCManager() {
        // 默认配置
    }

    void setSCdistThres(double thres) {
        manager_.setSCdistThres(thres);
    }

    void setMaximumRadius(double max_r) {
        manager_.setMaximumRadius(max_r);
    }

    double getSCdistThres() {
        return manager_.SC_DIST_THRES;
    }

    double getMaximumRadius() {
        return manager_.PC_MAX_RADIUS;
    }

    /**
     * 生成并保存ScanContext描述子
     * @param scan_down: 下采样后的点云 (Nx3 numpy array)
     */
    void makeAndSaveScancontextAndKeys(py::array_t<float> scan_down) {
        pcl::PointCloud<SCPointType>::Ptr cloud = numpy_to_pcl(scan_down);
        manager_.makeAndSaveScancontextAndKeys(*cloud);
    }

    /**
     * 检测回环
     * @return tuple(nearest_node_idx, relative_yaw)
     *         nearest_node_idx < 0 表示没有检测到回环
     */
    py::tuple detectLoopClosureID() {
        auto result = manager_.detectLoopClosureID();
        int nearest_node_idx = result.first;
        float relative_yaw = result.second;
        return py::make_tuple(nearest_node_idx, relative_yaw);
    }

    /**
     * 获取最近的ScanContext描述子
     * @return Eigen::MatrixXd (20x60 for default config)
     */
    Eigen::MatrixXd getRecentSCD() {
        return manager_.getConstRefRecentSCD();
    }

    /**
     * 获取数据库中的ScanContext数量
     */
    size_t getDatabaseSize() {
        return manager_.polarcontexts_.size();
    }

    /**
     * 获取配置参数
     */
    py::dict getConfig() {
        py::dict config;
        config["sc_dist_thres"] = manager_.SC_DIST_THRES;
        config["pc_max_radius"] = manager_.PC_MAX_RADIUS;
        config["pc_num_ring"] = manager_.PC_NUM_RING;
        config["pc_num_sector"] = manager_.PC_NUM_SECTOR;
        config["num_exclude_recent"] = manager_.NUM_EXCLUDE_RECENT;
        config["num_candidates_from_tree"] = manager_.NUM_CANDIDATES_FROM_TREE;
        config["search_ratio"] = manager_.SEARCH_RATIO;
        config["lidar_height"] = manager_.LIDAR_HEIGHT;
        return config;
    }
};

PYBIND11_MODULE(sc_cpp, m) {
    m.doc() = "ScanContext Python binding for loop closure detection";

    py::class_<PySCManager>(m, "SCManager")
        .def(py::init<>())
        .def("setSCdistThres", &PySCManager::setSCdistThres,
            "设置ScanContext距离阈值 (default: 0.6)")
        .def("setMaximumRadius", &PySCManager::setMaximumRadius,
            "设置ScanContext最大半径 (default: 80.0m)")
        .def("getSCdistThres", &PySCManager::getSCdistThres,
            "获取当前距离阈值")
        .def("getMaximumRadius", &PySCManager::getMaximumRadius,
            "获取当前最大半径")
        .def("makeAndSaveScancontextAndKeys", &PySCManager::makeAndSaveScancontextAndKeys,
            "生成并保存ScanContext描述子",
            py::arg("scan_down"))
        .def("detectLoopClosureID", &PySCManager::detectLoopClosureID,
            "检测回环，返回 (nearest_node_idx, relative_yaw)")
        .def("getRecentSCD", &PySCManager::getRecentSCD,
            "获取最近的ScanContext描述子")
        .def("getDatabaseSize", &PySCManager::getDatabaseSize,
            "获取数据库中的ScanContext数量")
        .def("getConfig", &PySCManager::getConfig,
            "获取配置参数");

    // 添加版本信息
    m.attr("__version__") = "1.0.0";
}