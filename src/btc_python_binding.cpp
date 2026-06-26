/**
 * BTC Python Binding - 通过pybind11将C++ BtcDescManager暴露给Python
 * 用于离线回环检测脚本直接调用C++实现
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "btc/btc.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace py = pybind11;

// ==================== 辅助函数: NumPy数组转PCL点云 ====================
pcl::PointCloud<pcl::PointXYZI>::Ptr numpy_to_pcl(py::array_t<float> array) {
    auto buf = array.request();
    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::runtime_error("Input array must be Nx4 (x, y, z, intensity)");
    }

    auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(buf.shape[0]);

    float* ptr = static_cast<float*>(buf.ptr);
    for (ssize_t i = 0; i < buf.shape[0]; i++) {  // 修复：使用ssize_t避免符号比较警告
        pcl::PointXYZI pt;
        pt.x = ptr[i * 4 + 0];
        pt.y = ptr[i * 4 + 1];
        pt.z = ptr[i * 4 + 2];
        pt.intensity = ptr[i * 4 + 3];
        cloud->push_back(pt);
    }

    return cloud;
}

// ==================== 辅助函数: PCL点云转NumPy数组 ====================
py::array_t<float> pcl_to_numpy(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    size_t n = cloud->size();
    
    // 修复：正确创建NumPy数组 (n, 4)
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n), 4};
    py::array_t<float> array(shape);
    auto buf = array.request();
    float* ptr = static_cast<float*>(buf.ptr);

    for (size_t i = 0; i < n; i++) {
        ptr[i * 4 + 0] = cloud->points[i].x;
        ptr[i * 4 + 1] = cloud->points[i].y;
        ptr[i * 4 + 2] = cloud->points[i].z;
        ptr[i * 4 + 3] = cloud->points[i].intensity;
    }

    return array;
}

// ==================== Python包装类 ====================
class PyBtcDescManager {
public:
    BtcDescManager manager_;
    std::string config_file_;  // 存储配置文件路径
    bool use_default_config_;  // 是否使用默认配置

    // 构造函数1: 不加载配置文件，使用内置默认值
    PyBtcDescManager() : use_default_config_(true) {
        // ConfigSetting已经在btc.h中设置好通用适配默认值，无需额外加载
        std::cout << "[BTC] 使用内置默认配置（通用适配参数）" << std::endl;
    }

    // 构造函数2: 加载配置文件，覆盖默认值
    PyBtcDescManager(std::string config_file) : config_file_(config_file), use_default_config_(false) {
        load_config_setting(config_file_, manager_.config_setting_);
    }

    // GenerateBtcDescs - 返回BTC数量和基本信息
    py::dict GenerateBtcDescs(py::array_t<float> cloud_array, int frame_id) {
        auto cloud = numpy_to_pcl(cloud_array);
        std::vector<BTC> btcs_vec;
        manager_.GenerateBtcDescs(cloud, frame_id, btcs_vec);

        py::dict result;
        result["num_btcs"] = btcs_vec.size();
        result["frame_id"] = frame_id;

        // 返回BTC三角形信息（用于调试）
        py::list triangles;
        for (const auto& btc : btcs_vec) {
            py::dict tri;
            tri["triangle"] = btc.triangle_;
            tri["center"] = btc.center_;
            tri["frame_number"] = btc.frame_number_;
            triangles.append(tri);
        }
        result["triangles"] = triangles;

        return result;
    }

    // SearchLoop - 返回匹配结果
    py::dict SearchLoop(py::array_t<float> cloud_array, int frame_id) {
        auto cloud = numpy_to_pcl(cloud_array);
        std::vector<BTC> btcs_vec;
        manager_.GenerateBtcDescs(cloud, frame_id, btcs_vec);

        std::pair<int, double> loop_result;
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        std::vector<std::pair<BTC, BTC>> loop_std_pair;

        manager_.SearchLoop(btcs_vec, loop_result, loop_transform, loop_std_pair);

        py::dict result;
        result["match_frame_id"] = loop_result.first;
        result["match_score"] = loop_result.second;
        result["translation"] = loop_transform.first;
        result["rotation"] = loop_transform.second;
        result["num_matches"] = loop_std_pair.size();

        return result;
    }

    // AddBtcDescs - 添加描述子到数据库
    void AddBtcDescs(py::array_t<float> cloud_array, int frame_id) {
        auto cloud = numpy_to_pcl(cloud_array);
        std::vector<BTC> btcs_vec;
        manager_.GenerateBtcDescs(cloud, frame_id, btcs_vec);
        manager_.AddBtcDescs(btcs_vec);
    }

    // 获取配置参数
    py::dict GetConfig() {
        py::dict config;
        config["voxel_size"] = manager_.config_setting_.voxel_size_;
        config["voxel_init_num"] = manager_.config_setting_.voxel_init_num_;
        config["summary_min_thre"] = manager_.config_setting_.summary_min_thre_;
        config["proj_plane_num"] = manager_.config_setting_.proj_plane_num_;
        config["descriptor_near_num"] = manager_.config_setting_.descriptor_near_num_;
        config["skip_near_num"] = manager_.config_setting_.skip_near_num_;
        config["useful_corner_num"] = manager_.config_setting_.useful_corner_num_;
        return config;
    }

    // 获取数据库大小
    size_t GetDatabaseSize() {
        return manager_.data_base_.size();
    }

    // 获取历史关键帧数
    size_t GetHistoryKeyframes() {
        return manager_.key_cloud_vec_.size();
    }

    // 开启/关闭C++侧详细调试日志（平面检测、合并率、描述子数等）
    void SetDebugInfo(bool enable) {
        manager_.print_debug_info_ = enable;
        std::cout << "[BTC] C++ debug info " << (enable ? "ENABLED" : "DISABLED") << std::endl;
    }
};

// ==================== pybind11模块定义 ====================
PYBIND11_MODULE(btc_cpp, m) {
    m.doc() = "BTC (Binary Triangle Context) C++ implementation for loop closure detection";

    py::class_<PyBtcDescManager>(m, "BtcDescManager")
        .def(py::init<>(), "默认构造：使用内置通用适配默认值")
        .def(py::init<std::string>(), py::arg("config_file"), "加载配置文件覆盖默认值")
        .def("GenerateBtcDescs", &PyBtcDescManager::GenerateBtcDescs,
             "Generate BTC descriptors from point cloud",
             py::arg("cloud"), py::arg("frame_id"))
        .def("SearchLoop", &PyBtcDescManager::SearchLoop,
             "Search for loop closure candidates",
             py::arg("cloud"), py::arg("frame_id"))
        .def("AddBtcDescs", &PyBtcDescManager::AddBtcDescs,
             "Add BTC descriptors to database",
             py::arg("cloud"), py::arg("frame_id"))
        .def("GetConfig", &PyBtcDescManager::GetConfig,
             "Get BTC configuration parameters")
        .def("GetDatabaseSize", &PyBtcDescManager::GetDatabaseSize,
             "Get size of BTC database")
        .def("GetHistoryKeyframes", &PyBtcDescManager::GetHistoryKeyframes,
             "Get number of history keyframes")
        .def("SetDebugInfo", &PyBtcDescManager::SetDebugInfo,
             "Enable/disable detailed C++ debug logging",
             py::arg("enable"));

    // 导出配置加载函数（修复：去掉const）
    m.def("load_config", [](std::string config_file) {
        ConfigSetting config;
        load_config_setting(config_file, config);
        py::dict result;
        result["voxel_size"] = config.voxel_size_;
        result["voxel_init_num"] = config.voxel_init_num_;
        result["summary_min_thre"] = config.summary_min_thre_;
        result["proj_plane_num"] = config.proj_plane_num_;
        return result;
    }, "Load BTC configuration from YAML file");

    // 导出下采样函数
    m.def("downsample_voxel", [](py::array_t<float> cloud_array, double voxel_size) {
        auto cloud = numpy_to_pcl(cloud_array);
        pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        down_sampling_voxel(*ds_cloud, voxel_size);
        return pcl_to_numpy(ds_cloud);
    }, "Downsample point cloud using voxel grid",
       py::arg("cloud"), py::arg("voxel_size"));
}