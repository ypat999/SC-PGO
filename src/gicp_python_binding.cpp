/**
 * GICP_OMP Python Binding - 通过pybind11将pclomp::GeneralizedIterativeClosestPoint暴露给Python
 * 用于离线回环检测脚本直接调用 C++ pclomp GICP（在线版本同款实现）
 *
 * 与 Open3D GICP 的关键区别：
 *   - pclomp GICP 是 on-line 用的同款实现，fitness_score 量级一致
 *   - 协方差计算用 OpenMP 并行，配准时也对 KDTree nearestKSearch 并行化
 *   - 接受 Nx3 (XYZ) 或 Nx4 (XYZI) numpy 数组
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pclomp/gicp_omp.h>
#include <pclomp/gicp_omp_impl.hpp>
#include <pcl/registration/exceptions.h>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

// ==================== NaN安全的KdTree搜索类 ====================
// 继承 pcl::search::KdTree，重写 nearestKSearch 和 radiusSearch 虚函数，
// 在调用原始实现前检查查询点是否包含 NaN/Inf，避免 KDTree assertion 崩溃。
// 通过 setSearchMethodTarget/Source 注入到 GICP 中。
class NaNCheckedKdTree : public pcl::search::KdTree<pcl::PointXYZI> {
public:
    using Base = pcl::search::KdTree<pcl::PointXYZI>;
    using Ptr = std::shared_ptr<NaNCheckedKdTree>;

    NaNCheckedKdTree(bool sorted = true) : Base(sorted) {}

    int nearestKSearch(const pcl::PointXYZI& point, int k,
                       pcl::Indices& k_indices,
                       std::vector<float>& k_sqr_distances) const override {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            k_indices.clear();
            k_sqr_distances.clear();
            return 0;
        }
        return Base::nearestKSearch(point, k, k_indices, k_sqr_distances);
    }

    int radiusSearch(const pcl::PointXYZI& point, double radius,
                     pcl::Indices& k_indices,
                     std::vector<float>& k_sqr_distances,
                     unsigned int max_nn = 0) const override {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            k_indices.clear();
            k_sqr_distances.clear();
            return 0;
        }
        return Base::radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);
    }
};

// ==================== 辅助函数: NumPy数组转PCL点云 ====================
pcl::PointCloud<pcl::PointXYZI>::Ptr numpy_to_pcl(py::array_t<float> array) {
    auto buf = array.request();
    int dim = (buf.ndim == 2) ? buf.shape[1] : 0;
    if (buf.ndim != 2 || (dim != 3 && dim != 4)) {
        throw std::runtime_error("Input array must be Nx3 or Nx4 (x, y, z, [intensity])");
    }

    auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(buf.shape[0]);

    float* ptr = static_cast<float*>(buf.ptr);
    for (ssize_t i = 0; i < buf.shape[0]; i++) {
        pcl::PointXYZI pt;
        pt.x = ptr[i * dim + 0];
        pt.y = ptr[i * dim + 1];
        pt.z = ptr[i * dim + 2];
        pt.intensity = (dim == 4) ? ptr[i * dim + 3] : 0.0f;
        cloud->push_back(pt);
    }
    return cloud;
}

// ==================== 辅助函数: 移除NaN/Inf点 ====================
static pcl::PointCloud<pcl::PointXYZI>::Ptr removeInvalidPoints(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
    auto clean = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    clean->reserve(cloud->size());
    for (const auto& p : cloud->points) {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            clean->push_back(p);
        }
    }
    return clean;
}

// ==================== GICP_OMP Python 包装类 ====================
class PyGicpOmpManager {
public:
    using GicpOmp = pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>;

    PyGicpOmpManager() {
        gicp_ = GicpOmp::Ptr(new GicpOmp());
        applyConfig();
        std::cout << "[GICP_OMP] 初始化完成 (pclomp), num_threads=" << num_threads_ << std::endl;
    }

    // ==================== 配置参数设置 ====================

    void setMaxCorrespondenceDistance(double dist) {
        corr_dist_threshold_ = dist;
        gicp_->setMaxCorrespondenceDistance(dist);
    }

    void setTransformationEpsilon(double eps) {
        transformation_epsilon_ = eps;
        gicp_->setTransformationEpsilon(eps);
    }

    void setRotationEpsilon(double eps) {
        rotation_epsilon_ = eps;
        gicp_->setRotationEpsilon(eps);
    }

    void setCorrespondenceRandomness(int k) {
        k_correspondences_ = k;
        gicp_->setCorrespondenceRandomness(k);
    }

    void setMaximumOptimizerIterations(int n) {
        max_optimizer_iterations_ = n;
        gicp_->setMaximumOptimizerIterations(n);
    }

    void setGICPEpsilon(double eps) {
        gicp_epsilon_ = eps;
        gicp_->setGICPEpsilon(eps);
    }

    void setMaximumIterations(int n) {
        max_iterations_ = n;
        gicp_->setMaximumIterations(n);
    }

    void setNumThreads(int n) {
        num_threads_ = n;
    }

    // 批量设置所有参数
    void setConfig(py::dict config) {
        if (config.contains("max_correspondence_distance"))
            setMaxCorrespondenceDistance(config["max_correspondence_distance"].cast<double>());
        if (config.contains("transformation_epsilon"))
            setTransformationEpsilon(config["transformation_epsilon"].cast<double>());
        if (config.contains("rotation_epsilon"))
            setRotationEpsilon(config["rotation_epsilon"].cast<double>());
        if (config.contains("k_correspondences"))
            setCorrespondenceRandomness(config["k_correspondences"].cast<int>());
        if (config.contains("max_optimizer_iterations"))
            setMaximumOptimizerIterations(config["max_optimizer_iterations"].cast<int>());
        if (config.contains("gicp_epsilon"))
            setGICPEpsilon(config["gicp_epsilon"].cast<double>());
        if (config.contains("max_iterations"))
            setMaximumIterations(config["max_iterations"].cast<int>());
        if (config.contains("num_threads"))
            setNumThreads(config["num_threads"].cast<int>());
    }

    // ==================== 配准接口 ====================

    py::dict align(py::array_t<float> source_array, py::array_t<float> target_array,
                   py::array_t<double> initial_guess_array = py::array_t<double>()) {
        auto source_cloud = numpy_to_pcl(source_array);
        auto target_cloud = numpy_to_pcl(target_array);

        // 解析初始猜测 (4x4 double)
        Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
        if (initial_guess_array.size() > 0) {
            auto buf = initial_guess_array.request();
            if (buf.ndim == 2 && buf.shape[0] == 4 && buf.shape[1] == 4) {
                double* ptr = static_cast<double*>(buf.ptr);
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        initial_guess(r, c) = static_cast<float>(ptr[r * 4 + c]);
            }
        }

        double init_t = initial_guess.block<3,1>(0,3).norm();
        std::cout << "[GICP_OMP] src=" << source_cloud->size()
                  << ", tgt=" << target_cloud->size()
                  << ", init_t=" << init_t << "m" << std::endl;

        gicp_->setInputSource(source_cloud);
        gicp_->setInputTarget(target_cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        try {
            gicp_->align(*aligned_cloud, initial_guess);
        } catch (const std::exception &e) {
            std::cout << "[GICP_OMP] 配准异常: " << e.what() << std::endl;
            py::dict result;
            result["transformation"] = initial_guess.cast<double>();
            result["has_converged"] = false;
            result["fitness_score"] = 1e9;
            return result;
        }

        Eigen::Matrix4f final_T = gicp_->getFinalTransformation();
        bool converged = gicp_->hasConverged();
        double fitness = gicp_->getFitnessScore();

        Eigen::Matrix4d T_double = final_T.cast<double>();
        double final_disp = T_double.block<3,1>(0,3).norm();
        std::cout << "[GICP_OMP] converged=" << (converged ? "true" : "false")
                  << ", fitness(overlap)=" << fitness
                  << ", final_disp=" << final_disp << "m" << std::endl;

        py::dict result;
        result["transformation"] = T_double;
        result["has_converged"] = converged;
        result["fitness_score"] = fitness;
        return result;
    }

    // 两级配准：先粗配准(下采样)，再精配准
    py::dict alignTwoStage(py::array_t<float> source_array, py::array_t<float> target_array,
                           py::array_t<double> initial_guess_array,
                           double coarse_ds_size, int coarse_max_iter, double coarse_max_dist,
                           int fine_max_iter, double fine_max_dist) {
        auto source_cloud = numpy_to_pcl(source_array);
        auto target_cloud = numpy_to_pcl(target_array);

        // 移除NaN/Inf点
        source_cloud = removeInvalidPoints(source_cloud);
        target_cloud = removeInvalidPoints(target_cloud);

        if (source_cloud->empty() || target_cloud->empty()) {
            std::cout << "[GICP_OMP] All input points invalid, skip" << std::endl;
            py::dict result;
            result["transformation"] = Eigen::Matrix4d::Identity();
            result["has_converged"] = false;
            result["fitness_score"] = 1e9;
            return result;
        }

        // 解析初始猜测
        Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
        if (initial_guess_array.size() > 0) {
            auto buf = initial_guess_array.request();
            if (buf.ndim == 2 && buf.shape[0] == 4 && buf.shape[1] == 4) {
                double* ptr = static_cast<double*>(buf.ptr);
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        initial_guess(r, c) = static_cast<float>(ptr[r * 4 + c]);
            }
        }

        double init_t = initial_guess.block<3,1>(0,3).norm();
        std::cout << "[GICP_OMP] src=" << source_cloud->size()
                  << ", tgt=" << target_cloud->size()
                  << ", init_t=" << init_t << "m" << std::endl;

        if (!initial_guess.allFinite()) {
            std::cout << "[GICP_OMP] 初始猜测矩阵包含NaN/Inf，跳过配准" << std::endl;
            py::dict result;
            result["transformation"] = Eigen::Matrix4d::Identity();
            result["has_converged"] = false;
            result["fitness_score"] = 1e9;
            return result;
        }

        // ===== 阶段1: 粗配准 (下采样) =====
        auto src_ds = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        auto tgt_ds = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        {
            pcl::VoxelGrid<pcl::PointXYZI> voxel;
            voxel.setLeafSize(coarse_ds_size, coarse_ds_size, coarse_ds_size);
            voxel.setInputCloud(source_cloud);
            voxel.filter(*src_ds);
            voxel.setInputCloud(target_cloud);
            voxel.filter(*tgt_ds);
        }
        src_ds = removeInvalidPoints(src_ds);
        tgt_ds = removeInvalidPoints(tgt_ds);

        std::cout << "[GICP_OMP] 粗配准: " << src_ds->size() << " vs " << tgt_ds->size()
                  << " pts (ds=" << coarse_ds_size << "m)" << std::endl;

        auto coarse_gicp = GicpOmp::Ptr(new GicpOmp());
        coarse_gicp->setTransformationEpsilon(transformation_epsilon_);
        coarse_gicp->setMaxCorrespondenceDistance(coarse_max_dist);
        coarse_gicp->setRotationEpsilon(rotation_epsilon_);
        coarse_gicp->setCorrespondenceRandomness(k_correspondences_);
        coarse_gicp->setMaximumOptimizerIterations(max_optimizer_iterations_);
        coarse_gicp->setGICPEpsilon(gicp_epsilon_);
        coarse_gicp->setMaximumIterations(coarse_max_iter);
        coarse_gicp->setSearchMethodTarget(std::make_shared<NaNCheckedKdTree>());
        coarse_gicp->setSearchMethodSource(std::make_shared<NaNCheckedKdTree>());
        coarse_gicp->setInputSource(src_ds);
        coarse_gicp->setInputTarget(tgt_ds);

        pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>);
        try {
            coarse_gicp->align(*tmp, initial_guess);
        } catch (const std::exception &e) {
            std::cout << "[GICP_OMP] 粗配准异常: " << e.what() << std::endl;
            py::dict result;
            result["transformation"] = initial_guess.cast<double>();
            result["has_converged"] = false;
            result["fitness_score"] = 1e9;
            return result;
        }

        Eigen::Matrix4f T_coarse = coarse_gicp->getFinalTransformation();
        std::cout << "[GICP_OMP] 粗配准: converged=" << (coarse_gicp->hasConverged() ? "true" : "false")
                  << ", fitness=" << coarse_gicp->getFitnessScore() << std::endl;

        // ===== 阶段2: 精配准 =====
        auto fine_gicp = GicpOmp::Ptr(new GicpOmp());
        fine_gicp->setTransformationEpsilon(transformation_epsilon_);
        fine_gicp->setMaxCorrespondenceDistance(fine_max_dist);
        fine_gicp->setRotationEpsilon(rotation_epsilon_);
        fine_gicp->setCorrespondenceRandomness(k_correspondences_);
        fine_gicp->setMaximumOptimizerIterations(max_optimizer_iterations_);
        fine_gicp->setGICPEpsilon(gicp_epsilon_);
        fine_gicp->setMaximumIterations(fine_max_iter);
        fine_gicp->setSearchMethodTarget(std::make_shared<NaNCheckedKdTree>());
        fine_gicp->setSearchMethodSource(std::make_shared<NaNCheckedKdTree>());
        fine_gicp->setInputSource(source_cloud);
        fine_gicp->setInputTarget(target_cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);
        try {
            fine_gicp->align(*output, T_coarse);
        } catch (const std::exception &e) {
            std::cout << "[GICP_OMP] 精配准异常: " << e.what() << std::endl;
            py::dict result;
            result["transformation"] = T_coarse.cast<double>();
            result["has_converged"] = false;
            result["fitness_score"] = 1e9;
            return result;
        }

        Eigen::Matrix4f final_T = fine_gicp->getFinalTransformation();
        bool converged = fine_gicp->hasConverged();

        double fitness = fine_gicp->getFitnessScore(1.0);
        double fitness_all = fine_gicp->getFitnessScore();

        // 计算overlap ratio（<=1m内点比例）
        double overlap_ratio = 0.0;
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr src_trans(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::transformPointCloud(*source_cloud, *src_trans, final_T);
            auto overlap_kdtree = std::make_shared<NaNCheckedKdTree>();
            overlap_kdtree->setInputCloud(target_cloud);
            int inlier_count = 0;
            for (size_t i = 0; i < src_trans->size(); ++i) {
                pcl::Indices idx(1);
                std::vector<float> dist(1);
                if (overlap_kdtree->nearestKSearch(src_trans->points[i], 1, idx, dist) > 0
                    && dist[0] < 1.0f) {
                    inlier_count++;
                }
            }
            overlap_ratio = static_cast<double>(inlier_count) / src_trans->size();
        }

        Eigen::Matrix4d T_double = final_T.cast<double>();
        double final_disp = T_double.block<3,1>(0,3).norm();
        std::cout << "[GICP_OMP] 精配准: converged=" << (converged ? "true" : "false")
                  << ", fitness(1m)=" << fitness
                  << ", fitness(all)=" << fitness_all
                  << ", overlap(1m)=" << (overlap_ratio * 100) << "%"
                  << ", final_disp=" << final_disp << "m" << std::endl;

        py::dict result;
        result["transformation"] = T_double;
        result["has_converged"] = converged;
        result["fitness_score"] = fitness;
        result["overlap_ratio"] = overlap_ratio;
        return result;
    }

    // ==================== 配置查询 ====================
    py::dict getConfig() {
        py::dict c;
        c["max_correspondence_distance"] = corr_dist_threshold_;
        c["transformation_epsilon"] = transformation_epsilon_;
        c["rotation_epsilon"] = rotation_epsilon_;
        c["k_correspondences"] = k_correspondences_;
        c["max_optimizer_iterations"] = max_optimizer_iterations_;
        c["gicp_epsilon"] = gicp_epsilon_;
        c["max_iterations"] = max_iterations_;
        c["num_threads"] = num_threads_;
        return c;
    }

private:
    GicpOmp::Ptr gicp_;
    int num_threads_ = 4;

    double corr_dist_threshold_ = 5.0;
    double transformation_epsilon_ = 5e-4;
    double rotation_epsilon_ = 2e-3;
    int k_correspondences_ = 20;
    int max_optimizer_iterations_ = 20;
    double gicp_epsilon_ = 0.001;
    int max_iterations_ = 200;

    void applyConfig() {
        gicp_->setMaxCorrespondenceDistance(corr_dist_threshold_);
        gicp_->setTransformationEpsilon(transformation_epsilon_);
        gicp_->setRotationEpsilon(rotation_epsilon_);
        gicp_->setCorrespondenceRandomness(k_correspondences_);
        gicp_->setMaximumOptimizerIterations(max_optimizer_iterations_);
        gicp_->setGICPEpsilon(gicp_epsilon_);
        gicp_->setMaximumIterations(max_iterations_);
    }
};

// ==================== pybind11 模块定义 ====================
PYBIND11_MODULE(gicp_omp_cpp, m) {
    m.doc() = "pclomp::GICP_OMP Python binding — 同款 on-line 配准算法";

    py::class_<PyGicpOmpManager>(m, "GicpOmpManager")
        .def(py::init<>(), "默认构造，使用与在线版本一致的默认参数")
        .def("setConfig", &PyGicpOmpManager::setConfig,
             "批量设置GICP参数", py::arg("config"))
        .def("setMaxCorrespondenceDistance", &PyGicpOmpManager::setMaxCorrespondenceDistance)
        .def("setTransformationEpsilon", &PyGicpOmpManager::setTransformationEpsilon)
        .def("setRotationEpsilon", &PyGicpOmpManager::setRotationEpsilon)
        .def("setCorrespondenceRandomness", &PyGicpOmpManager::setCorrespondenceRandomness)
        .def("setMaximumOptimizerIterations", &PyGicpOmpManager::setMaximumOptimizerIterations)
        .def("setGICPEpsilon", &PyGicpOmpManager::setGICPEpsilon)
        .def("setMaximumIterations", &PyGicpOmpManager::setMaximumIterations)
        .def("setNumThreads", &PyGicpOmpManager::setNumThreads)
        .def("align", &PyGicpOmpManager::align,
             "执行GICP配准",
             py::arg("source"), py::arg("target"),
             py::arg("initial_guess") = py::array_t<double>())
        .def("alignTwoStage", &PyGicpOmpManager::alignTwoStage,
             "两级GICP：先下采样粗配准，再原分辨率精配准",
             py::arg("source"), py::arg("target"), py::arg("initial_guess"),
             py::arg("coarse_ds_size") = 0.3,
             py::arg("coarse_max_iter") = 50,
             py::arg("coarse_max_dist") = 3.0,
             py::arg("fine_max_iter") = 200,
             py::arg("fine_max_dist") = 2.0)
        .def("getConfig", &PyGicpOmpManager::getConfig,
             "获取当前GICP配置参数");
}
