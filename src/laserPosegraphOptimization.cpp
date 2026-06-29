#include <math.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
// #include <pcl/search/impl/search.hpp>
// #include <pcl/range_image/range_image.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/common/common.h>
#include <pcl/common/transforms.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
// #include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/octree/octree_pointcloud_voxelcentroid.h>
// #include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// #include <sensor_msgs/Imu.h>
// #include <tf/transform_datatypes.h>
// #include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// #include <eigen3/Eigen/Dense>

// #include <ceres/ceres.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "btc/btc.h"
#include "gicp_registration/gicp_registration.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0;
double rotaionAccumulated = 1000000.0;

bool isNowKeyFrame = false;

Pose6D odom_pose_prev{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // init
Pose6D odom_pose_curr{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // init pose is zero

std::queue<std::shared_ptr<nav_msgs::msg::Odometry>> odometryBuf;
std::queue<std::shared_ptr<sensor_msgs::msg::PointCloud2>> fullResBuf;
std::queue<std::shared_ptr<sensor_msgs::msg::NavSatFix>> gpsBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(
    new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(
    new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
BtcDescManager btcManager;
double scDistThres, scMaximumRadius;

std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(
    new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::msg::NavSatFix::SharedPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false;
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftPGO;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPathAftPGO;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubOptimizedPath;  // PGO optimized path (same as pubPathAftPGO, different topic name)
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubOdomPath;       // raw odom keyframe trajectory (for PGO comparison)
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapAftPGO;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLoopScanLocal;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLoopSubmapLocal;

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopMatchMarkers;  // loop match points + connecting line

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomRepubVerifier;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

std::shared_ptr<rclcpp::Node> nh;

std::string frame_id_odom;
std::string frame_id_aft_pgo;

// GICP registration
sc_pgo::GICPRegistration* gicp_registration;
bool use_gicp_for_loop_closure = true;
double gicp_fitness_score_threshold = 0.25;
double gicp_max_init_translation = 15.0;

// Loop closure validation
double max_loop_distance = 100.0;
double max_yaw_diff = M_PI * 0.75;

// Odom Direct verification
double odom_direct_threshold = 3.0;  // odom距离<此值时跳过BTC直接GICP验证

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D &p) {
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw),
                      gtsam::Point3(p.x, p.y, p.z));
}  // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename) {
  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), std::fstream::out);
  for (const auto &_pose6d : keyframePoses) {
    gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1);  // Point3
    auto col2 = R.column(2);  // Point3
    auto col3 = R.column(3);  // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x()
           << " " << col1.y() << " " << col2.y() << " " << col3.y() << " "
           << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z()
           << " " << t.z() << std::endl;
  }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates,
                                      std::string _filename) {
  using namespace gtsam;

  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), std::fstream::out);

  for (const auto &key_value : _estimates) {
    auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
    if (!p) continue;

    const Pose3 &pose = p->value();

    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1);  // Point3
    auto col2 = R.column(2);  // Point3
    auto col3 = R.column(3);  // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x()
           << " " << col1.y() << " " << col2.y() << " " << col3.y() << " "
           << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z()
           << " " << t.z() << std::endl;
  }
}

void laserOdometryHandler(
    const nav_msgs::msg::Odometry::SharedPtr _laserOdometry) {
  mBuf.lock();
  odometryBuf.push(_laserOdometry);
  mBuf.unlock();
}  // laserOdometryHandler

void laserCloudFullResHandler(
    sensor_msgs::msg::PointCloud2::SharedPtr _laserCloudFullRes) {
  mBuf.lock();
  fullResBuf.push(_laserCloudFullRes);
  mBuf.unlock();
}  // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::msg::NavSatFix::SharedPtr _gps) {
  if (useGPS) {
    mBuf.lock();
    gpsBuf.push(_gps);
    mBuf.unlock();
  }
}  // gpsHandler

void initNoises(void) {
  gtsam::Vector priorNoiseVector6(6);
  priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
  priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

  gtsam::Vector odomNoiseVector6(6);
  // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
  odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
  odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

  double loopNoiseScore = 0.5;  // constant is ok...
  gtsam::Vector robustNoiseVector6(
      6);  // gtsam::Pose3 factor has 6 elements (6D)
  robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
      loopNoiseScore, loopNoiseScore, loopNoiseScore;
  robustLoopNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(
          1.345),  // Huber kernel is more robust than Cauchy for loop closures
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

  double bigNoiseTolerentToXY = 1000000000.0;  // 1e9
  double gpsAltitudeNoiseScore = 250.0;  // if height is misaligned after loop
                                         // clsosing, use this value bigger
  gtsam::Vector robustNoiseVector3(3);   // gps factor has 3 elements (xyz)
  robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY,
      gpsAltitudeNoiseScore;  // means only caring altitude here. (because
                              // LOAM-like-methods tends to be asymptotically
                              // flyging)
  robustGPSNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(
          1),  // optional: replacing Cauchy by DCS or GemanMcClure is okay but
               // Cauchy is empirically good.
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3));
}  // initNoises

Pose6D getOdom(const nav_msgs::msg::Odometry::SharedPtr &_odom) {
  auto tx = _odom->pose.pose.position.x;
  auto ty = _odom->pose.pose.position.y;
  auto tz = _odom->pose.pose.position.z;

  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion quat = _odom->pose.pose.orientation;
  tf2::Quaternion q(quat.x, quat.y, quat.z, quat.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(roll, pitch, yaw);
  return Pose6D{tx, ty, tz, roll, pitch, yaw};
}  // getOdom

Pose6D diffTransformation(const Pose6D &_p1, const Pose6D &_p2) {
  Eigen::Affine3f SE3_p1 =
      pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
  Eigen::Affine3f SE3_p2 =
      pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
  Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
  Eigen::Affine3f SE3_delta;
  SE3_delta.matrix() = SE3_delta0;
  float dx, dy, dz, droll, dpitch, dyaw;
  pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
  // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll
  // << ", " << dpitch << ", " << dyaw << std::endl;

  return Pose6D{double(abs(dx)),    double(abs(dy)),     double(abs(dz)),
                double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
}  // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(
    const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D &tf) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur =
      pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

  int numberOfCores = 16;
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i) {
    const auto &pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                            transCur(0, 1) * pointFrom.y +
                            transCur(0, 2) * pointFrom.z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                            transCur(1, 1) * pointFrom.y +
                            transCur(1, 2) * pointFrom.z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                            transCur(2, 1) * pointFrom.y +
                            transCur(2, 2) * pointFrom.z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }

  return cloudOut;
}

void pubPath(void) {
  // Publish odom and path
  nav_msgs::msg::Odometry odomAftPGO;
  nav_msgs::msg::Path pathAftPGO;
  nav_msgs::msg::Path pathOdom;  // raw odom keyframe trajectory (before PGO)
  pathAftPGO.header.frame_id = frame_id_odom;
  pathOdom.header.frame_id = frame_id_odom;
  mKF.lock();
  for (int node_idx = 0; node_idx < recentIdxUpdated; node_idx++) {
    const Pose6D &pose_est =
        keyframePosesUpdated.at(node_idx);  // Updated poses
    const Pose6D &pose_odom =
        keyframePoses.at(node_idx);  // Raw odom keyframe poses

    nav_msgs::msg::Odometry odomAftPGOthis;
    odomAftPGOthis.header.frame_id = frame_id_odom;
    odomAftPGOthis.child_frame_id = frame_id_aft_pgo;
    odomAftPGOthis.header.stamp =
        rclcpp::Time(keyframeTimes.at(node_idx) * 1e9);
    odomAftPGOthis.pose.pose.position.x = pose_est.x;
    odomAftPGOthis.pose.pose.position.y = pose_est.y;
    odomAftPGOthis.pose.pose.position.z = pose_est.z;

    tf2::Quaternion q;
    q.setRPY(pose_est.roll, pose_est.pitch, pose_est.yaw);
    odomAftPGOthis.pose.pose.orientation.x = q.x();
    odomAftPGOthis.pose.pose.orientation.y = q.y();
    odomAftPGOthis.pose.pose.orientation.z = q.z();
    odomAftPGOthis.pose.pose.orientation.w = q.w();
    odomAftPGO = odomAftPGOthis;

    geometry_msgs::msg::PoseStamped poseStampAftPGO;
    poseStampAftPGO.header = odomAftPGOthis.header;
    poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

    pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
    pathAftPGO.header.frame_id = frame_id_odom;
    pathAftPGO.poses.push_back(poseStampAftPGO);

    // raw odom path pose
    geometry_msgs::msg::PoseStamped poseStampOdom;
    poseStampOdom.header = odomAftPGOthis.header;
    poseStampOdom.pose.position.x = pose_odom.x;
    poseStampOdom.pose.position.y = pose_odom.y;
    poseStampOdom.pose.position.z = pose_odom.z;
    tf2::Quaternion q_odom;
    q_odom.setRPY(pose_odom.roll, pose_odom.pitch, pose_odom.yaw);
    poseStampOdom.pose.orientation.x = q_odom.x();
    poseStampOdom.pose.orientation.y = q_odom.y();
    poseStampOdom.pose.orientation.z = q_odom.z();
    poseStampOdom.pose.orientation.w = q_odom.w();
    pathOdom.header.stamp = odomAftPGOthis.header.stamp;
    pathOdom.header.frame_id = frame_id_odom;
    pathOdom.poses.push_back(poseStampOdom);
  }
  mKF.unlock();
  pubOdomAftPGO->publish(odomAftPGO);  // Last pose
  pubPathAftPGO->publish(pathAftPGO);  // Optimized poses
  pubOptimizedPath->publish(pathAftPGO); // Same optimized path on "optimized_path" topic
  pubOdomPath->publish(pathOdom);      // Raw odom keyframe poses

  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped.header.stamp = odomAftPGO.header.stamp;
  transformStamped.header.frame_id = frame_id_odom;
  transformStamped.child_frame_id = frame_id_aft_pgo;
  transformStamped.transform.translation.x = odomAftPGO.pose.pose.position.x;
  transformStamped.transform.translation.y = odomAftPGO.pose.pose.position.y;
  transformStamped.transform.translation.z = odomAftPGO.pose.pose.position.z;
  transformStamped.transform.rotation = odomAftPGO.pose.pose.orientation;

  static std::shared_ptr<tf2_ros::TransformBroadcaster> br = nullptr;
  if (!br) {
    br = std::make_shared<tf2_ros::TransformBroadcaster>(nh);
  }
  br->sendTransform(transformStamped);
}  // pubPath

// Publish loop match markers: two spheres (prev/curr keyframe positions) + a line connecting them.
// Uses keyframePosesUpdated so the markers line up with the optimized path shown in RViz.
void publishLoopMatchMarkers(int prev_idx, int curr_idx, double score) {
  if (!pubLoopMatchMarkers) return;

  mKF.lock();
  if (prev_idx < 0 || prev_idx >= int(keyframePosesUpdated.size()) ||
      curr_idx < 0 || curr_idx >= int(keyframePosesUpdated.size())) {
    mKF.unlock();
    return;
  }
  Pose6D p_prev = keyframePosesUpdated[prev_idx];
  Pose6D p_curr = keyframePosesUpdated[curr_idx];
  mKF.unlock();

  visualization_msgs::msg::MarkerArray marker_array;
  rclcpp::Time now = nh->now();

  // Use a persistent namespace for all loop markers; id is unique per loop event
  // (encoded as prev_idx * 100000 + curr_idx to avoid collisions).
  int base_id = prev_idx * 100000 + curr_idx;

  // Marker 1: prev keyframe point (red sphere)
  visualization_msgs::msg::Marker m_prev;
  m_prev.header.frame_id = frame_id_odom;
  m_prev.header.stamp = now;
  m_prev.ns = "loop_match_points";
  m_prev.id = base_id;
  m_prev.type = visualization_msgs::msg::Marker::SPHERE;
  m_prev.action = visualization_msgs::msg::Marker::ADD;
  m_prev.pose.position.x = p_prev.x;
  m_prev.pose.position.y = p_prev.y;
  m_prev.pose.position.z = p_prev.z;
  m_prev.pose.orientation.w = 1.0;
  m_prev.scale.x = m_prev.scale.y = m_prev.scale.z = 0.6;  // 60cm sphere
  m_prev.color.r = 1.0f;
  m_prev.color.g = 0.0f;
  m_prev.color.b = 0.0f;
  m_prev.color.a = 1.0f;
  m_prev.lifetime = rclcpp::Duration(0, 0);  // persistent
  marker_array.markers.push_back(m_prev);

  // Marker 2: curr keyframe point (green sphere)
  visualization_msgs::msg::Marker m_curr = m_prev;
  m_curr.id = base_id + 1;
  m_curr.pose.position.x = p_curr.x;
  m_curr.pose.position.y = p_curr.y;
  m_curr.pose.position.z = p_curr.z;
  m_curr.color.r = 0.0f;
  m_curr.color.g = 1.0f;
  m_curr.color.b = 0.0f;
  marker_array.markers.push_back(m_curr);

  // Marker 3: connecting line (yellow LINE_LIST with 2 points)
  visualization_msgs::msg::Marker m_line;
  m_line.header.frame_id = frame_id_odom;
  m_line.header.stamp = now;
  m_line.ns = "loop_match_lines";
  m_line.id = base_id;
  m_line.type = visualization_msgs::msg::Marker::LINE_LIST;
  m_line.action = visualization_msgs::msg::Marker::ADD;
  m_line.scale.x = 0.05;  // line width 5cm
  m_line.color.r = 1.0f;
  m_line.color.g = 1.0f;
  m_line.color.b = 0.0f;
  m_line.color.a = 1.0f;
  m_line.pose.orientation.w = 1.0;
  geometry_msgs::msg::Point pt_prev, pt_curr;
  pt_prev.x = p_prev.x; pt_prev.y = p_prev.y; pt_prev.z = p_prev.z;
  pt_curr.x = p_curr.x; pt_curr.y = p_curr.y; pt_curr.z = p_curr.z;
  m_line.points.push_back(pt_prev);
  m_line.points.push_back(pt_curr);
  m_line.lifetime = rclcpp::Duration(0, 0);
  marker_array.markers.push_back(m_line);

  // Marker 4 (optional): text label showing loop pair indices and score
  visualization_msgs::msg::Marker m_text;
  m_text.header.frame_id = frame_id_odom;
  m_text.header.stamp = now;
  m_text.ns = "loop_match_labels";
  m_text.id = base_id;
  m_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  m_text.action = visualization_msgs::msg::Marker::ADD;
  m_text.pose.position.x = (p_prev.x + p_curr.x) * 0.5;
  m_text.pose.position.y = (p_prev.y + p_curr.y) * 0.5;
  m_text.pose.position.z = (p_prev.z + p_curr.z) * 0.5 + 1.0;
  m_text.pose.orientation.w = 1.0;
  m_text.scale.z = 0.5;  // text height
  m_text.color.r = 1.0f;
  m_text.color.g = 1.0f;
  m_text.color.b = 1.0f;
  m_text.color.a = 1.0f;
  std::ostringstream oss;
  oss << "loop " << prev_idx << "<->" << curr_idx << " s=" << std::fixed
      << std::setprecision(3) << score;
  m_text.text = oss.str();
  m_text.lifetime = rclcpp::Duration(0, 0);
  marker_array.markers.push_back(m_text);

  pubLoopMatchMarkers->publish(marker_array);
}

void updatePoses(void) {
  mKF.lock();
  for (int node_idx = 0; node_idx < int(isamCurrentEstimate.size());
       node_idx++) {
    Pose6D &p = keyframePosesUpdated[node_idx];
    p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
    p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
    p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
    p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
    p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
    p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
  }
  mKF.unlock();

  mtxRecentPose.lock();
  const gtsam::Pose3 &lastOptimizedPose =
      isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size()) - 1);
  recentOptimizedX = lastOptimizedPose.translation().x();
  recentOptimizedY = lastOptimizedPose.translation().y();

  recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

  mtxRecentPose.unlock();
}  // updatePoses

void runISAM2opt(void) {
  // called when a variable added
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  gtSAMgraph.resize(0);
  initialEstimate.clear();

  isamCurrentEstimate = isam->calculateEstimate();
  updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
      transformIn.translation().x(), transformIn.translation().y(),
      transformIn.translation().z(), transformIn.rotation().roll(),
      transformIn.rotation().pitch(), transformIn.rotation().yaw());

  int numberOfCores = 4;  // TODO move to yaml
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom->x +
                            transCur(0, 1) * pointFrom->y +
                            transCur(0, 2) * pointFrom->z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom->x +
                            transCur(1, 1) * pointFrom->y +
                            transCur(1, 2) * pointFrom->z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom->x +
                            transCur(2, 1) * pointFrom->y +
                            transCur(2, 2) * pointFrom->z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom->intensity;
  }
  return cloudOut;
}  // transformPointCloud

void loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
                                const int &key, const int &submap_size,
                                const int &root_idx) {
  // extract and stacking near keyframes (in global coord)
  nearKeyframes->clear();
  for (int i = -submap_size; i <= submap_size; ++i) {
    int keyNear = key + i;
    if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size())) continue;

    mKF.lock();
    *nearKeyframes += *local2global(keyframeLaserClouds[keyNear],
                                    keyframePosesUpdated[root_idx]);
    mKF.unlock();
  }

  if (nearKeyframes->empty()) return;

  // downsample near keyframes
  pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
  downSizeFilterScancontext.setInputCloud(nearKeyframes);
  downSizeFilterScancontext.filter(*cloud_temp);
  *nearKeyframes = *cloud_temp;
}  // loopFindNearKeyframesCloud

std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx,
                                                 int _curr_kf_idx,
                                                 float yaw_diff = 0.0f) {
  int historyKeyframeSearchNum =
      25;  // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every
           // kf gap is 1m
  pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(
      new pcl::PointCloud<PointType>());
  loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0,
                             _loop_kf_idx);  // use same root of loop kf idx
  loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx,
                             historyKeyframeSearchNum, _loop_kf_idx);

  // loop verification
  sensor_msgs::msg::PointCloud2 cureKeyframeCloudMsg;
  pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
  cureKeyframeCloudMsg.header.frame_id = frame_id_odom;
  pubLoopScanLocal->publish(cureKeyframeCloudMsg);

  sensor_msgs::msg::PointCloud2 targetKeyframeCloudMsg;
  pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
  targetKeyframeCloudMsg.header.frame_id = frame_id_odom;
  pubLoopSubmapLocal->publish(targetKeyframeCloudMsg);

  // Stage 1: Coarse ICP matching with relaxed parameters
  pcl::IterativeClosestPoint<PointType, PointType> icp_coarse;
  icp_coarse.setMaxCorrespondenceDistance(30.0);  // Reduced from 150 to 30 meters
  icp_coarse.setMaximumIterations(50);
  icp_coarse.setTransformationEpsilon(1e-4);
  icp_coarse.setEuclideanFitnessEpsilon(1e-4);
  icp_coarse.setRANSACIterations(0);

  // Use yaw difference from Scan Context as initial rotation guess
  Eigen::Affine3f initial_guess = Eigen::Affine3f::Identity();
  initial_guess.rotate(Eigen::AngleAxisf(yaw_diff, Eigen::Vector3f::UnitZ()));

  icp_coarse.setInputSource(cureKeyframeCloud);
  icp_coarse.setInputTarget(targetKeyframeCloud);
  pcl::PointCloud<PointType>::Ptr coarse_result(
      new pcl::PointCloud<PointType>());
  icp_coarse.align(*coarse_result, initial_guess.matrix());

  float coarseFitnessThreshold = 1.5;
  if (icp_coarse.hasConverged() == false ||
      icp_coarse.getFitnessScore() > coarseFitnessThreshold) {
    std::cout << "[SC loop] Coarse ICP failed (" << icp_coarse.getFitnessScore()
              << " > " << coarseFitnessThreshold << "). Reject this SC loop."
              << std::endl;
    return std::nullopt;
  }

  std::cout << "[SC loop] Coarse ICP passed (" << icp_coarse.getFitnessScore()
            << " < " << coarseFitnessThreshold << "). Proceeding to fine ICP."
            << std::endl;

  // Stage 2: Fine ICP matching with strict parameters
  pcl::IterativeClosestPoint<PointType, PointType> icp_fine;
  icp_fine.setMaxCorrespondenceDistance(2.0);  // Strict correspondence distance
  icp_fine.setMaximumIterations(100);
  icp_fine.setTransformationEpsilon(1e-6);
  icp_fine.setEuclideanFitnessEpsilon(1e-6);
  icp_fine.setRANSACIterations(0);

  icp_fine.setInputSource(cureKeyframeCloud);
  icp_fine.setInputTarget(targetKeyframeCloud);
  pcl::PointCloud<PointType>::Ptr unused_result(
      new pcl::PointCloud<PointType>());
  icp_fine.align(*unused_result, icp_coarse.getFinalTransformation());

  float loopFitnessScoreThreshold = 0.5;
  if (icp_fine.hasConverged() == false ||
      icp_fine.getFitnessScore() > loopFitnessScoreThreshold) {
    std::cout << "[SC loop] Fine ICP fitness test failed (" << icp_fine.getFitnessScore()
              << " > " << loopFitnessScoreThreshold << "). Reject this SC loop."
              << std::endl;
    return std::nullopt;
  } else {
    std::cout << "[SC loop] Fine ICP fitness test passed (" << icp_fine.getFitnessScore()
              << " < " << loopFitnessScoreThreshold << "). Add this SC loop."
              << std::endl;
  }

  // Get pose transformation from fine ICP
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame = icp_fine.getFinalTransformation();
  pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch,
                                    yaw);
  gtsam::Pose3 poseFrom =
      Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
  gtsam::Pose3 poseTo =
      Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  return poseFrom.between(poseTo);
}  // doICPVirtualRelative

template <typename PointT>
void removeNaNAndInfiniteInPlace(typename pcl::PointCloud<PointT>::Ptr &cloud) {
  if (!cloud || cloud->empty()) return;

  // First pass: remove NaNs using PCL’s built-in
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  // Second pass: remove infinities in-place
  size_t write_idx = 0;
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    const auto &pt = cloud->points[i];
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
      cloud->points[write_idx++] = pt;
    }
  }

  cloud->points.resize(write_idx);
  cloud->width = static_cast<uint32_t>(write_idx);
  cloud->height = 1;
  cloud->is_dense = true;
}

void process_pg() {
  static int frame_counter = 0;
  while (1) {
    while (!odometryBuf.empty() && !fullResBuf.empty()) {
      frame_counter++;
      
      //
      // pop and check keyframe is or not
      //
      // cout << "=== Process PG Frame " << frame_counter << " ===" << endl;
      // cout << "Odometry buffer size: " << odometryBuf.size() << endl;
      // cout << "FullRes buffer size: " << fullResBuf.size() << endl;
      
      mBuf.lock();
      while (!odometryBuf.empty() &&
             rclcpp::Time(odometryBuf.front()->header.stamp).seconds() <
                 rclcpp::Time(fullResBuf.front()->header.stamp).seconds())
        odometryBuf.pop();
      if (odometryBuf.empty()) {
        cout << "Odometry buffer empty after sync, breaking..." << endl;
        mBuf.unlock();
        break;
      }

      // Time equal check
      timeLaserOdometry =
          rclcpp::Time(odometryBuf.front()->header.stamp).seconds();
      timeLaser = rclcpp::Time(fullResBuf.front()->header.stamp).seconds();
      // cout << "Time check - Odometry: " << timeLaserOdometry << ", Laser: " << timeLaser << endl;
      // TODO

      laserCloudFullRes->clear();
      pcl::PointCloud<PointType>::Ptr thisKeyFrame(
          new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
      fullResBuf.pop();

      Pose6D pose_curr = getOdom(odometryBuf.front());
      odometryBuf.pop();

      // find nearest gps
      double eps = 0.1;  // find a gps topioc arrived within eps second
      // cout << "GPS buffer size: " << gpsBuf.size() << endl;
      while (!gpsBuf.empty()) {
        auto thisGPS = gpsBuf.front();
        auto thisGPSTime = rclcpp::Time(thisGPS->header.stamp).seconds();
        double time_diff = abs(thisGPSTime - timeLaserOdometry);
        // cout << "GPS time diff: " << time_diff << " (threshold: " << eps << ")" << endl;
        if (time_diff < eps) {
          currGPS = thisGPS;
          hasGPSforThisKF = true;
          // cout << "GPS found for this keyframe, altitude: " << currGPS->altitude << endl;
          break;
        } else {
          hasGPSforThisKF = false;
        }
        gpsBuf.pop();
      }
      if (!hasGPSforThisKF) {
        // cout << "No GPS found for this keyframe" << endl;
      }
      mBuf.unlock();

      //
      // Early reject by counting local delta movement (for equi-spereated kf
      // drop)
      //
      odom_pose_prev = odom_pose_curr;
      odom_pose_curr = pose_curr;
      Pose6D dtf = diffTransformation(
          odom_pose_prev, odom_pose_curr);  // dtf means delta_transform

      double delta_translation = sqrt(dtf.x * dtf.x + dtf.y * dtf.y +
                                      dtf.z * dtf.z);  // note: absolute value.
      translationAccumulated += delta_translation;
      rotaionAccumulated +=
          (dtf.roll + dtf.pitch + dtf.yaw);  // sum just naive approach.

      // cout << "Delta movement - Translation: " << delta_translation 
      //      << ", Accumulated: " << translationAccumulated 
      //      << " (threshold: " << keyframeMeterGap << ")" << endl;
      // cout << "Delta rotation - Accumulated: " << rotaionAccumulated 
      //      << " (threshold: " << keyframeRadGap << ")" << endl;

      if (translationAccumulated > keyframeMeterGap ||
          rotaionAccumulated > keyframeRadGap) {
        isNowKeyFrame = true;
        translationAccumulated = 0.0;  // reset
        rotaionAccumulated = 0.0;      // reset
        cout << "Keyframe detected!" << endl;
      } else {
        isNowKeyFrame = false;
        // cout << "Not a keyframe, skipping..." << endl;
      }

      if (!isNowKeyFrame) continue;

      if (!gpsOffsetInitialized) {
        if (hasGPSforThisKF) {  // if the very first frame
          gpsAltitudeInitOffset = currGPS->altitude;
          gpsOffsetInitialized = true;
        }
      }

      //
      // Save data and Add consecutive node
      //
      // cout << "Processing keyframe data..." << endl;
      pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(
          new pcl::PointCloud<PointType>());
      downSizeFilterScancontext.setInputCloud(thisKeyFrame);
      downSizeFilterScancontext.filter(*thisKeyFrameDS);
      removeNaNAndInfiniteInPlace<PointType>(thisKeyFrameDS);

      cout << "Original keyframe points: " << thisKeyFrame->size() 
           << ", Downsampled: " << thisKeyFrameDS->size() << endl;

      mKF.lock();
      keyframeLaserClouds.push_back(thisKeyFrameDS);
      keyframePoses.push_back(pose_curr);
      keyframePosesUpdated.push_back(pose_curr);  // init
      keyframeTimes.push_back(timeLaserOdometry);

      cout << "Current keyframe count: " << keyframePoses.size() << endl;
      cout << "Keyframe pose - x: " << pose_curr.x << ", y: " << pose_curr.y 
           << ", z: " << pose_curr.z << ", roll: " << pose_curr.roll 
           << ", pitch: " << pose_curr.pitch << ", yaw: " << pose_curr.yaw << endl;

      int curr_frame_id = keyframePoses.size() - 1;
      std::vector<BTC> btcs_vec;
      pcl::PointCloud<pcl::PointXYZI>::Ptr btcCloud(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud(*thisKeyFrameDS, *btcCloud);
      cout << "[BTC] Starting GenerateBtcDescs for frame " << curr_frame_id
           << ", cloud points: " << btcCloud->size() << endl;
      try {
        btcManager.GenerateBtcDescs(btcCloud, curr_frame_id, btcs_vec);
        cout << "[BTC] GenerateBtcDescs done, btcs count: " << btcs_vec.size() << endl;
      } catch (const std::exception &e) {
        std::cerr << "[BTC] GenerateBtcDescs exception: " << e.what() << std::endl;
        throw;
      }
      btcManager.AddBtcDescs(btcs_vec);
      cout << "[BTC] AddBtcDescs done" << endl;

      laserCloudMapPGORedraw = true;
      mKF.unlock();

      const int prev_node_idx = keyframePoses.size() - 2;
      const int curr_node_idx =
          keyframePoses.size() -
          1;  // becuase cpp starts with 0 (actually this index could be any
              // number, but for simple implementation, we follow sequential
              // indexing)
      
      // cout << "Adding to posegraph - Prev node: " << prev_node_idx 
      //      << ", Curr node: " << curr_node_idx << endl;
      
      if (!gtSAMgraphMade /* prior node */) {
        const int init_node_idx = 0;
        gtsam::Pose3 poseOrigin =
            Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
        // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0),
        // gtsam::Point3(0.0, 0.0, 0.0));

        // cout << "Adding prior node " << init_node_idx << " to posegraph" << endl;
        
        mtxPosegraph.lock();
        {
          // prior factor
          gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(
              init_node_idx, poseOrigin, priorNoise));
          initialEstimate.insert(init_node_idx, poseOrigin);
          // runISAM2opt();
        }
        mtxPosegraph.unlock();

        gtSAMgraphMade = true;

        // cout << "posegraph prior node " << init_node_idx << " added" << endl;
      } else /* consecutive node (and odom factor) after the prior added */
      {      // == keyframePoses.size() > 1
        // cout << "Adding consecutive node " << curr_node_idx << " to posegraph" << endl;
        
        gtsam::Pose3 poseFrom =
            Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
        gtsam::Pose3 poseTo =
            Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

        // cout << "Pose from node " << prev_node_idx << " to node " << curr_node_idx << endl;
        
        mtxPosegraph.lock();
        {
          // odom factor
          gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
              prev_node_idx, curr_node_idx, poseFrom.between(poseTo),
              odomNoise));
          // cout << "Odom factor added between nodes " << prev_node_idx << " and " << curr_node_idx << endl;

          // gps factor
          if (hasGPSforThisKF) {
            double curr_altitude_offseted =
                currGPS->altitude - gpsAltitudeInitOffset;
            mtxRecentPose.lock();
            gtsam::Point3 gpsConstraint(
                recentOptimizedX, recentOptimizedY,
                curr_altitude_offseted);  // in this example, only adjusting
                                          // altitude (for x and y, very big
                                          // noises are set)
            mtxRecentPose.unlock();
            gtSAMgraph.add(
                gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
            // cout << "GPS factor added at node " << curr_node_idx 
            //      << ", altitude offset: " << curr_altitude_offseted << endl;
          } else {
            // cout << "No GPS factor added for node " << curr_node_idx << endl;
          }
          initialEstimate.insert(curr_node_idx, poseTo);
          // runISAM2opt();
        }
        mtxPosegraph.unlock();

        // if (curr_node_idx % 100 == 0) {
        //   cout << "posegraph odom node " << curr_node_idx << " added." << endl;
        // } else {
        //   cout << "Node " << curr_node_idx << " added to posegraph" << endl;
        // }
      }
      // if want to print the current graph, use gtSAMgraph.print("\nFactor
      // Graph:\n");

      // save utility
      std::string curr_node_idx_str = padZeros(curr_node_idx);
      std::string pcd_filename = pgScansDirectory + curr_node_idx_str + ".pcd";
      pcl::io::savePCDFileBinary(pcd_filename, *thisKeyFrame);   // scan
      pgTimeSaveStream << timeLaser << std::endl;  // path
      
      // cout << "Saved keyframe data to: " << pcd_filename << endl;
      // cout << "=== Process PG Frame " << frame_counter << " Completed ===" << endl;
      // cout << endl;  // Add empty line for better readability
    }

    // ps.
    // scan context detector is running in another thread (in constant Hz, e.g.,
    // 1 Hz) pub path and point cloud in another thread

    // wait (must required for running the while loop)
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
    
    if (frame_counter > 0 && frame_counter % 1000 == 0) {
      cout << "Process PG still running... Total frames processed: " << frame_counter << endl;
      cout << "Odometry buffer size: " << odometryBuf.size() << endl;
      cout << "FullRes buffer size: " << fullResBuf.size() << endl;
    }
  }
}  // process_pg

bool validateLoopClosure(int prev_idx, int curr_idx, gtsam::Pose3 relative_pose);

void performSCLoopClosure(void) {
  if (int(keyframePoses.size()) < 20)
    return;

  int curr_frame_id = keyframePoses.size() - 1;

  // ===== Odom Direct 验证: odom距离<阈值时直接GICP，跳过BTC =====
  if (use_gicp_for_loop_closure && odom_direct_threshold > 0) {
    mKF.lock();
    Pose6D curr_pose = keyframePoses[curr_frame_id];
    auto curr_cloud = keyframeLaserClouds[curr_frame_id];
    mKF.unlock();

    int skip_near = btcManager.config_setting_.skip_near_num_;
    for (int j = 0; j < curr_frame_id; j++) {
      if ((curr_frame_id - j) <= skip_near) continue;  // 跳过邻近帧

      mKF.lock();
      Pose6D prev_pose = keyframePoses[j];
      auto prev_cloud = keyframeLaserClouds[j];
      mKF.unlock();

      double dx = curr_pose.x - prev_pose.x;
      double dy = curr_pose.y - prev_pose.y;
      double dz = curr_pose.z - prev_pose.z;
      double odom_dist = sqrt(dx*dx + dy*dy + dz*dz);

      if (odom_dist < odom_direct_threshold) {
        // 计算 odom 相对位姿初值
        gtsam::Pose3 pose_prev_gtsam = Pose6DtoGTSAMPose3(prev_pose);
        gtsam::Pose3 pose_curr_gtsam = Pose6DtoGTSAMPose3(curr_pose);
        gtsam::Pose3 odom_relative = pose_prev_gtsam.between(pose_curr_gtsam);
        Eigen::Matrix4d init_guess = odom_relative.matrix();

        double init_t = init_guess.block<3, 1>(0, 3).norm();
        if (init_t > gicp_max_init_translation) {
          cout << "[Odom Direct] frame " << j << " odom_dist=" << odom_dist
               << "m, but init_t=" << init_t << "m > " << gicp_max_init_translation
               << "m, skip" << endl;
          continue;
        }

        cout << "[Odom Direct] frame " << curr_frame_id << " <-> " << j
             << " odom_dist=" << odom_dist << "m < " << odom_direct_threshold
             << "m, trying GICP..." << endl;

        sc_pgo::GICPResult gicp_result = gicp_registration->align(
          curr_cloud, prev_cloud, init_guess
        );

        if (gicp_result.has_converged &&
            gicp_result.fitness_score < gicp_fitness_score_threshold) {
          gtsam::Pose3 relative_pose(gicp_result.transformation.cast<double>());

          if (validateLoopClosure(j, curr_frame_id, relative_pose)) {
            mtxPosegraph.lock();
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                j, curr_frame_id, relative_pose, robustLoopNoise));
            mtxPosegraph.unlock();
            cout << "[Odom Direct] constraint added between " << j
                 << " and " << curr_frame_id << " (fitness="
                 << gicp_result.fitness_score << ")" << endl;
            // publish loop match markers (two points + connecting line)
            publishLoopMatchMarkers(j, curr_frame_id, gicp_result.fitness_score);
            return;  // 找到一个就返回，跳过BTC流程
          }
        } else {
          cout << "[Odom Direct] GICP failed (fitness="
               << gicp_result.fitness_score << "), continue searching" << endl;
        }
      }
    }
  }

  // ===== BTC 回环检测（正常流程） =====
  std::vector<BTC> btcs_vec;
  pcl::PointCloud<pcl::PointXYZI>::Ptr btcCloud(new pcl::PointCloud<pcl::PointXYZI>);

  mKF.lock();
  auto currKeyDS = keyframeLaserClouds[curr_frame_id];
  mKF.unlock();

  pcl::copyPointCloud(*currKeyDS, *btcCloud);
  btcManager.GenerateBtcDescs(btcCloud, curr_frame_id, btcs_vec);

  std::pair<int, double> loop_result(-1, 0);
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
  std::vector<std::pair<BTC, BTC>> loop_std_pair;
  btcManager.SearchLoop(btcs_vec, loop_result, loop_transform, loop_std_pair);

  int BTCclosestHistoryFrameID = loop_result.first;
  double loopScore = loop_result.second;

  if (BTCclosestHistoryFrameID != -1) {
    const int prev_node_idx = BTCclosestHistoryFrameID;
    const int curr_node_idx = keyframePoses.size() - 1;
    cout << "[BTC Loop] detected! - between " << prev_node_idx << " and "
         << curr_node_idx << ", score: " << loopScore << endl;

    Eigen::Matrix3d rot = loop_transform.second;
    Eigen::Vector3d t = loop_transform.first;
    Eigen::Matrix4d relative_pose_matrix = Eigen::Matrix4d::Identity();
    relative_pose_matrix.block<3, 3>(0, 0) = rot;
    relative_pose_matrix.block<3, 1>(0, 3) = t;
    gtsam::Pose3 relative_pose(relative_pose_matrix.cast<double>());

    // Use GICP for refinement if enabled
    if (use_gicp_for_loop_closure) {
      // 检查初始平移是否过大（避免GICP崩溃）
      double init_translation = relative_pose_matrix.block<3, 1>(0, 3).norm();
      if (init_translation > gicp_max_init_translation) {
        std::cout << "[GICP] Initial translation " << init_translation 
                  << "m > " << gicp_max_init_translation << "m, skipping GICP refinement." << std::endl;
      } else {
        mKF.lock();
        auto currCloud = keyframeLaserClouds[curr_node_idx];
        auto prevCloud = keyframeLaserClouds[prev_node_idx];
        mKF.unlock();

        // GICP refinement using BTC result as initial guess
        sc_pgo::GICPResult gicp_result = gicp_registration->align(
          currCloud, prevCloud, relative_pose_matrix.cast<double>()
        );

        if (gicp_result.has_converged && 
            gicp_result.fitness_score < gicp_fitness_score_threshold) {
          // Use GICP refined pose
          relative_pose = gtsam::Pose3(gicp_result.transformation.cast<double>());
          cout << "[GICP] Refinement successful! Fitness score: " 
               << gicp_result.fitness_score << endl;
        } else {
          cout << "[GICP] Refinement failed or score too high (" 
               << gicp_result.fitness_score << "), using BTC result" << endl;
        }
      }
    }

    if (validateLoopClosure(prev_node_idx, curr_node_idx, relative_pose)) {
      mtxPosegraph.lock();
      gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
      mtxPosegraph.unlock();
      cout << "[BTC Loop] constraint added between " << prev_node_idx
           << " and " << curr_node_idx << endl;
      // publish loop match markers (two points + connecting line)
      publishLoopMatchMarkers(prev_node_idx, curr_node_idx, loopScore);
    }
  }
}  // performSCLoopClosure

void process_lcd() {
  float loopClosureFrequency = 1.0;
  rclcpp::Rate rate(loopClosureFrequency);

  while (rclcpp::ok()) {
    rate.sleep();
    performSCLoopClosure();
  }
}

bool validateLoopClosure(int prev_idx, int curr_idx, gtsam::Pose3 relative_pose) {
  mKF.lock();
  gtsam::Pose3 pose_prev = Pose6DtoGTSAMPose3(keyframePoses[prev_idx]);
  gtsam::Pose3 pose_curr = Pose6DtoGTSAMPose3(keyframePoses[curr_idx]);
  mKF.unlock();
  
  gtsam::Pose3 pose_diff = pose_prev.between(pose_curr);
  
  double distance = pose_diff.translation().norm();
  if (distance > max_loop_distance) {
    std::cout << "[Loop validation] Distance too large: " << distance 
              << " > " << max_loop_distance << ". Reject loop." << std::endl;
    return false;
  }
  
  double yaw_diff_val = std::abs(pose_diff.rotation().yaw());
  if (yaw_diff_val > max_yaw_diff) {
    std::cout << "[Loop validation] Yaw difference too large: " << yaw_diff_val 
              << " > " << max_yaw_diff << ". Reject loop." << std::endl;
    return false;
  }
  
  std::cout << "[Loop validation] Passed. Distance: " << distance 
            << ", Yaw diff: " << yaw_diff_val << std::endl;
  return true;
}

// process_icp removed: BTC's SearchLoop already includes planar geometric ICP,
// so the separate ICP verification thread is no longer needed.
// doICPVirtualRelative is kept for potential future use.

void process_viz_path() {
  float hz = 10.0;
  rclcpp::Rate rate(hz);

  while (rclcpp::ok()) {
    rate.sleep();
    if (recentIdxUpdated > 1) {
      pubPath();
    }
  }
}

void process_isam() {
  float hz = 1.0;
  rclcpp::Rate rate(hz);

  while (rclcpp::ok()) {
    rate.sleep();
    if (gtSAMgraphMade) {
      mtxPosegraph.lock();
      runISAM2opt();
      std::cout << "running isam2 optimization ..." << std::endl;
      mtxPosegraph.unlock();

      saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat);
      saveOdometryVerticesKITTIformat(odomKITTIformat);
    }
  }
}

void pubMap(void) {
  int SKIP_FRAMES = 2;  // sparse map visulalization to save computations
  int counter = 0;

  laserCloudMapPGO->clear();

  mKF.lock();
  // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size());
  // node_idx++) {
  for (int node_idx = 0; node_idx < recentIdxUpdated; node_idx++) {
    if (counter % SKIP_FRAMES == 0) {
      *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx],
                                         keyframePosesUpdated[node_idx]);
    }
    counter++;
  }
  mKF.unlock();

  downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
  downSizeFilterMapPGO.filter(*laserCloudMapPGO);

  sensor_msgs::msg::PointCloud2 laserCloudMapPGOMsg;
  pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
  laserCloudMapPGOMsg.header.frame_id = frame_id_odom;
  pubMapAftPGO->publish(laserCloudMapPGOMsg);
}

void process_viz_map() {
  float vizmapFrequency = 0.1;  // Hz, so one cycle every 10 seconds
  rclcpp::Rate rate(vizmapFrequency);

  while (rclcpp::ok()) {
    rate.sleep();
    if (recentIdxUpdated > 1) {
      pubMap();
    }
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("laserPGO");

  nh->declare_parameter<std::string>("save_directory", "/");
  save_directory = nh->get_parameter("save_directory").as_string();

  nh->declare_parameter<std::string>("frame_id_odom", "odom");
  frame_id_odom = nh->get_parameter("frame_id_odom").as_string();

  nh->declare_parameter<std::string>("frame_id_aft_pgo", "aft_pgo");
  frame_id_aft_pgo = nh->get_parameter("frame_id_aft_pgo").as_string();

  // prepend namespace to frame_ids if not already present
  std::string ns = nh->get_namespace();
  if (ns != "/" && ns != "") {
    if (frame_id_odom.find("/") == std::string::npos) {
      frame_id_odom = ns + "/" + frame_id_odom;
    }
    if (frame_id_aft_pgo.find("/") == std::string::npos) {
      frame_id_aft_pgo = ns + "/" + frame_id_aft_pgo;
    }
  }

  // GICP parameters
  nh->declare_parameter<bool>("use_gicp_for_loop_closure", true);
  use_gicp_for_loop_closure = nh->get_parameter("use_gicp_for_loop_closure").as_bool();

  nh->declare_parameter<double>("gicp_fitness_score_threshold", 0.25);
  gicp_fitness_score_threshold = nh->get_parameter("gicp_fitness_score_threshold").as_double();

  // Initialize GICP registration
  sc_pgo::GICPConfig gicp_config;
  nh->declare_parameter<double>("gicp_transformation_epsilon", 0.001);
  gicp_config.transformation_epsilon = nh->get_parameter("gicp_transformation_epsilon").as_double();

  nh->declare_parameter<double>("gicp_max_correspondence_distance", 5.0);
  gicp_config.max_correspondence_distance = nh->get_parameter("gicp_max_correspondence_distance").as_double();

  nh->declare_parameter<int>("gicp_max_iterations", 64);
  gicp_config.max_iterations = nh->get_parameter("gicp_max_iterations").as_int();

  nh->declare_parameter<int>("gicp_num_threads", 4);
  gicp_config.num_threads = nh->get_parameter("gicp_num_threads").as_int();

  nh->declare_parameter<double>("gicp_max_init_translation", 15.0);
  gicp_config.max_init_translation = nh->get_parameter("gicp_max_init_translation").as_double();
  gicp_max_init_translation = gicp_config.max_init_translation;

  // Two-stage alignment parameters
  nh->declare_parameter<double>("gicp_scan_ds_size", 0.1);
  gicp_config.scan_ds_size = nh->get_parameter("gicp_scan_ds_size").as_double();

  nh->declare_parameter<double>("gicp_coarse_ds_size", 0.25);
  gicp_config.coarse_ds_size = nh->get_parameter("gicp_coarse_ds_size").as_double();

  nh->declare_parameter<int>("gicp_coarse_max_iter", 50);
  gicp_config.coarse_max_iter = nh->get_parameter("gicp_coarse_max_iter").as_int();

  nh->declare_parameter<double>("gicp_coarse_max_dist", 10.0);
  gicp_config.coarse_max_dist = nh->get_parameter("gicp_coarse_max_dist").as_double();

  gicp_registration = new sc_pgo::GICPRegistration(gicp_config);

  // Loop closure validation parameters
  nh->declare_parameter<double>("max_loop_distance", 100.0);
  max_loop_distance = nh->get_parameter("max_loop_distance").as_double();

  nh->declare_parameter<double>("max_yaw_diff", M_PI * 0.75);
  max_yaw_diff = nh->get_parameter("max_yaw_diff").as_double();

  // Odom Direct verification parameter
  nh->declare_parameter<double>("odom_direct_threshold", 3.0);
  odom_direct_threshold = nh->get_parameter("odom_direct_threshold").as_double();

  nh->declare_parameter<double>("keyframe_meter_gap", 2.0);
  keyframeMeterGap = nh->get_parameter("keyframe_meter_gap").as_double();

  nh->declare_parameter<double>("keyframe_deg_gap", 10.0);
  keyframeDegGap = nh->get_parameter("keyframe_deg_gap").as_double();

  odomKITTIformat = save_directory + "odom_poses.txt";
  pgKITTIformat = save_directory + "optimized_poses.txt";
  pgTimeSaveStream =
      std::fstream(save_directory + "times.txt", std::fstream::out);
  pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
  pgScansDirectory = save_directory + "Scans/";
  cout << "pgScansDirectory " << pgScansDirectory << endl;

  keyframeRadGap = deg2rad(keyframeDegGap);

  nh->declare_parameter<double>("sc_dist_thres", 0.2);
  scDistThres = nh->get_parameter("sc_dist_thres").as_double();

  nh->declare_parameter<double>(
      "sc_max_radius",
      20.0);
  scMaximumRadius = nh->get_parameter("sc_max_radius").as_double();

  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);
  initNoises();

  nh->declare_parameter<std::string>("btc_config_file", "");
  std::string btc_config_file = nh->get_parameter("btc_config_file").as_string();
  if (btc_config_file.empty()) {
    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("sc_pgo_ros2");
    btc_config_file = pkg_share_dir + "/config/btc_config.yaml";
  }
  ConfigSetting btc_config;
  load_config_setting(btc_config_file, btc_config);
  btcManager = BtcDescManager(btc_config);

  float filter_size = 0.4;
  downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);

  double mapVizFilterSize;
  nh->declare_parameter<double>("mapviz_filter_size", 0.4);
  mapVizFilterSize = nh->get_parameter("mapviz_filter_size")
                         .as_double();  // pose assignment every k frames
  downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize,
                                   mapVizFilterSize);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      subLaserCloudFullRes;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPS;

  subLaserCloudFullRes = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
      "velodyne_cloud_registered_local", rclcpp::SensorDataQoS(), laserCloudFullResHandler);

  subLaserOdometry = nh->create_subscription<nav_msgs::msg::Odometry>(
      "aft_mapped_to_init", 100, laserOdometryHandler);

  subGPS = nh->create_subscription<sensor_msgs::msg::NavSatFix>("/gps/fix", 100,
                                                                gpsHandler);

  pubOdomAftPGO = nh->create_publisher<nav_msgs::msg::Odometry>(
      "aft_pgo_odom", rclcpp::QoS(100));
  pubOdomRepubVerifier = nh->create_publisher<nav_msgs::msg::Odometry>(
      "repub_odom", rclcpp::QoS(100));
  pubPathAftPGO = nh->create_publisher<nav_msgs::msg::Path>("/aft_pgo_path",
                                                            rclcpp::QoS(100));
  pubOptimizedPath = nh->create_publisher<nav_msgs::msg::Path>(
      "optimized_path", rclcpp::QoS(100));  // PGO optimized path
  pubOdomPath = nh->create_publisher<nav_msgs::msg::Path>(
      "odom_keyframe_path", rclcpp::QoS(100));  // raw odom keyframe trajectory
  pubMapAftPGO = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
      "aft_pgo_map", rclcpp::QoS(100));

  pubLoopScanLocal = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
      "loop_scan_local", rclcpp::QoS(100));
  pubLoopSubmapLocal = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
      "loop_submap_local", rclcpp::QoS(100));

  pubLoopMatchMarkers = nh->create_publisher<visualization_msgs::msg::MarkerArray>(
      "loop_match_markers", rclcpp::QoS(100).transient_local());  // latched so late subscribers see all loops

  std::thread posegraph_slam{process_pg};  // pose graph construction
  std::thread lc_detection{process_lcd};   // loop closure detection
  // process_icp thread removed: BTC's SearchLoop already includes planar geometric ICP
  std::thread isam_update{
      process_isam};  // if you want to call less isam2 run (for saving
                      // redundant computations and no real-time visulization is
                      // required), uncommment this and comment all the above
                      // runisam2opt when node is added.

  std::thread viz_map{process_viz_map};  // visualization - map (low frequency
                                         // because it is heavy)
  std::thread viz_path{
      process_viz_path};  // visualization - path (high frequency)

  rclcpp::spin(nh);
  return 0;
}
