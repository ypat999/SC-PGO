#ifndef GICP_REGISTRATION_HPP
#define GICP_REGISTRATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pclomp/gicp_omp.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace sc_pgo {

typedef pcl::PointXYZI PointType;

struct GICPConfig {
  double transformation_epsilon = 1e-6;
  double max_correspondence_distance = 30.0;
  double rotation_epsilon = 0.002;
  int k_correspondences = 20;
  int max_optimizer_iterations = 20;
  double gicp_epsilon = 0.01;
  int max_iterations = 100;
  double fitness_score_threshold = 0.5;
  int num_threads = 4;
};

struct GICPResult {
  Eigen::Matrix4d transformation;
  bool has_converged;
  double fitness_score;
  int num_iterations;
};

class GICPRegistration {
public:
  GICPRegistration();
  GICPRegistration(const GICPConfig& config);
  
  void setConfig(const GICPConfig& config);
  
  GICPResult align(
    const pcl::PointCloud<PointType>::Ptr& source_cloud,
    const pcl::PointCloud<PointType>::Ptr& target_cloud,
    const Eigen::Matrix4d& initial_guess = Eigen::Matrix4d::Identity()
  );
  
private:
  GICPConfig config_;
  pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr gicp_;
  
  void initializeGICP();
};

} // namespace sc_pgo

#endif // GICP_REGISTRATION_HPP