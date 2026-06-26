#include "gicp_registration/gicp_registration.hpp"
#include <pclomp/gicp_omp_impl.hpp>
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
  
  gicp_->setInputSource(source_cloud);
  gicp_->setInputTarget(target_cloud);
  
  pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>);
  
  // Perform alignment
  gicp_->align(*aligned_cloud, initial_guess.cast<float>());
  
  // Get results
  result.transformation = gicp_->getFinalTransformation().cast<double>();
  result.has_converged = gicp_->hasConverged();
  result.fitness_score = gicp_->getFitnessScore();
  result.num_iterations = 0;  // not available in this GICP version
  
  std::cout << "[GICP] Converged: " << result.has_converged 
            << ", Fitness score: " << result.fitness_score << std::endl;
  
  return result;
}

} // namespace sc_pgo