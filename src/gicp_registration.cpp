#include "gicp_registration/gicp_registration.hpp"
#include <pclomp/gicp_omp_impl.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <iostream>

namespace sc_pgo {

GICPRegistration::GICPRegistration() {
  initializeGICP();
}

GICPRegistration::GICPRegistration(const GICPConfig& config)
  : config_(config) {
  initializeGICP();
}

void GICPRegistration::setConfig(const GICPConfig& config) {
  config_ = config;
  initializeGICP();
}

void GICPRegistration::initializeGICP() {
  gicp_ = pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr(
    new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>()
  );

  gicp_->setTransformationEpsilon(config_.transformation_epsilon);
  gicp_->setMaxCorrespondenceDistance(config_.max_correspondence_distance);
  gicp_->setRotationEpsilon(config_.rotation_epsilon);
  gicp_->setCorrespondenceRandomness(config_.k_correspondences);
  gicp_->setMaximumOptimizerIterations(config_.max_optimizer_iterations);
  gicp_->setGICPEpsilon(config_.gicp_epsilon);
  gicp_->setMaximumIterations(config_.max_iterations);
}

GICPResult GICPRegistration::align(
  const pcl::PointCloud<PointType>::Ptr& source_cloud,
  const pcl::PointCloud<PointType>::Ptr& target_cloud,
  const Eigen::Matrix4d& initial_guess
) {
  GICPResult result;
  result.transformation = initial_guess;
  result.has_converged = false;
  result.fitness_score = std::numeric_limits<double>::max();
  result.num_iterations = 0;

  if (!source_cloud || source_cloud->empty() ||
      !target_cloud || target_cloud->empty()) {
    std::cout << "[GICP] Empty input clouds" << std::endl;
    return result;
  }

  // ===== Stage 1: Coarse alignment (downsampled) =====
  pcl::PointCloud<PointType>::Ptr src_ds(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr tgt_ds(new pcl::PointCloud<PointType>);
  {
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(config_.coarse_ds_size, config_.coarse_ds_size, config_.coarse_ds_size);
    voxel.setInputCloud(source_cloud);
    voxel.filter(*src_ds);
    voxel.setInputCloud(target_cloud);
    voxel.filter(*tgt_ds);
  }

  std::cout << "[GICP] Coarse: " << src_ds->size() << " vs " << tgt_ds->size()
            << " pts (ds=" << config_.coarse_ds_size << "m)" << std::endl;

  auto coarse_gicp = pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr(
    new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>()
  );
  coarse_gicp->setTransformationEpsilon(config_.transformation_epsilon);
  coarse_gicp->setMaxCorrespondenceDistance(config_.coarse_max_dist);
  coarse_gicp->setRotationEpsilon(config_.rotation_epsilon);
  coarse_gicp->setCorrespondenceRandomness(config_.k_correspondences);
  coarse_gicp->setMaximumOptimizerIterations(config_.max_optimizer_iterations);
  coarse_gicp->setGICPEpsilon(config_.gicp_epsilon);
  coarse_gicp->setMaximumIterations(config_.coarse_max_iter);
  coarse_gicp->setInputSource(src_ds);
  coarse_gicp->setInputTarget(tgt_ds);

  pcl::PointCloud<PointType>::Ptr coarse_aligned(new pcl::PointCloud<PointType>);
  Eigen::Matrix4f init_f = initial_guess.cast<float>();
  coarse_gicp->align(*coarse_aligned, init_f);

  Eigen::Matrix4f T_coarse = coarse_gicp->getFinalTransformation();
  std::cout << "[GICP] Coarse: converged=" << (coarse_gicp->hasConverged() ? 1 : 0)
            << ", fitness=" << coarse_gicp->getFitnessScore() << std::endl;

  // ===== Stage 2: Fine alignment (original resolution) =====
  auto fine_gicp = pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr(
    new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>()
  );
  fine_gicp->setTransformationEpsilon(config_.transformation_epsilon);
  fine_gicp->setMaxCorrespondenceDistance(config_.max_correspondence_distance);
  fine_gicp->setRotationEpsilon(config_.rotation_epsilon);
  fine_gicp->setCorrespondenceRandomness(config_.k_correspondences);
  fine_gicp->setMaximumOptimizerIterations(config_.max_optimizer_iterations);
  fine_gicp->setGICPEpsilon(config_.gicp_epsilon);
  fine_gicp->setMaximumIterations(config_.max_iterations);
  fine_gicp->setInputSource(source_cloud);
  fine_gicp->setInputTarget(target_cloud);

  pcl::PointCloud<PointType>::Ptr fine_aligned(new pcl::PointCloud<PointType>);
  fine_gicp->align(*fine_aligned, T_coarse);

  Eigen::Matrix4f final_T = fine_gicp->getFinalTransformation();
  bool converged = fine_gicp->hasConverged();

  // Fitness score with 1.0m threshold (same as offline version)
  double fitness = fine_gicp->getFitnessScore(1.0);

  result.transformation = final_T.cast<double>();
  result.has_converged = converged;
  result.fitness_score = fitness;

  std::cout << "[GICP] Fine: converged=" << (converged ? 1 : 0)
            << ", fitness(1m)=" << fitness << std::endl;

  return result;
}

} // namespace sc_pgo
