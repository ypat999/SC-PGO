#include "btc/btc.h"

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }

  config_setting.useful_corner_num_ = fSettings["useful_corner_num"];
  config_setting.plane_merge_normal_thre_ =
      fSettings["plane_merge_normal_thre"];
  config_setting.plane_merge_dis_thre_ = fSettings["plane_merge_dis_thre"];
  config_setting.plane_detection_thre_ = fSettings["plane_detection_thre"];
  config_setting.voxel_size_ = fSettings["voxel_size"];
  config_setting.voxel_init_num_ = fSettings["voxel_init_num"];
  config_setting.proj_plane_num_ = fSettings["proj_plane_num"];
  config_setting.proj_image_resolution_ = fSettings["proj_image_resolution"];
  config_setting.proj_image_high_inc_ = fSettings["proj_image_high_inc"];
  config_setting.proj_dis_min_ = fSettings["proj_dis_min"];
  config_setting.proj_dis_max_ = fSettings["proj_dis_max"];
  config_setting.summary_min_thre_ = fSettings["summary_min_thre"];
  config_setting.line_filter_enable_ = fSettings["line_filter_enable"];

  config_setting.descriptor_near_num_ = fSettings["descriptor_near_num"];
  config_setting.descriptor_min_len_ = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len_ = fSettings["descriptor_max_len"];
  config_setting.non_max_suppression_radius_ = fSettings["non_max_suppression_radius"];
  config_setting.std_side_resolution_ = fSettings["triangle_resolution"];

  config_setting.skip_near_num_ = fSettings["skip_near_num"];
  config_setting.candidate_num_ = fSettings["candidate_num"];
  config_setting.rough_dis_threshold_ = fSettings["rough_dis_threshold"];
  config_setting.similarity_threshold_ = fSettings["similarity_threshold"];
  config_setting.icp_threshold_ = fSettings["icp_threshold"];
  config_setting.normal_threshold_ = fSettings["normal_threshold"];
  config_setting.dis_threshold_ = fSettings["dis_threshold"];
  config_setting.geom_side_std_threshold_ = fSettings["geom_side_std_threshold"];
  config_setting.geom_center_std_threshold_ = fSettings["geom_center_std_threshold"];
  config_setting.ransac_min_vote_ = fSettings["ransac_min_vote"];
  config_setting.ransac_max_iterations_ = fSettings["ransac_max_iterations"];
  config_setting.ransac_sample_max_ = fSettings["ransac_sample_max"];
  config_setting.ransac_correspondence_dis_ = fSettings["ransac_correspondence_dis"];
  config_setting.candidate_selector_min_vote_ = fSettings["candidate_selector_min_vote"];
  config_setting.candidate_verify_min_pairs_ = fSettings["candidate_verify_min_pairs"];
  config_setting.icp_point_to_point_max_ = fSettings["icp_point_to_point_max"];

  std::cout << "Sucessfully load config file:" << config_file << std::endl;
}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2) {
  // P1-2: 改用Jaccard相似度，增加惩罚项，避免descriptor太容易撞分
  double match = 0;      // 1-1 匹配（加分）
  double mismatch = 0;   // 1-0 或 0-1 不匹配（惩罚）

  for (size_t i = 0; i < b1.occupy_array_.size(); i++) {
    if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true) {
      match++;
    } else if (b1.occupy_array_[i] == true || b2.occupy_array_[i] == true) {
      mismatch++;  // 惩罚项：一方occupied另一方不occupied
    }
    // 0-0一致不计入，因为这是"共同空白"，不如"共同occupied"有价值
  }

  // Jaccard相似度：match / (match + mismatch)
  // 当match=0时返回0，避免除零
  if (match + mismatch == 0) {
    return 0.0;
  }
  return match / (match + mismatch);
}

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b) {
  return (a.summary_ > b.summary_);
}

bool plane_greater_sort(std::shared_ptr<Plane> plane1,
                        std::shared_ptr<Plane> plane2) {
  return plane1->points_size_ > plane2->points_size_;
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > config_setting_.voxel_init_num_) {
    init_plane();
  }
}

void OctoTree::init_plane() {
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                       plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                       plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  double mean_triangle_dis = 0;
  for (auto var : match_std_list) {
    mean_triangle_dis += (var.first.triangle_ - var.second.triangle_).norm() /
                         var.first.triangle_.norm();
  }
  if (match_std_list.size() > 0) {
    mean_triangle_dis = mean_triangle_dis / match_std_list.size();
  } else {
    mean_triangle_dis = -1;
  }
  return mean_triangle_dis;
}

double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  double mean_binary_similarity = 0;
  for (auto var : match_std_list) {
    mean_binary_similarity +=
        (binary_similarity(var.first.binary_A_, var.second.binary_A_) +
         binary_similarity(var.first.binary_B_, var.second.binary_B_) +
         binary_similarity(var.first.binary_C_, var.second.binary_C_)) /
        3;
  }
  if (match_std_list.size() > 0) {
    mean_binary_similarity = mean_binary_similarity / match_std_list.size();
  } else {
    mean_binary_similarity = -1;
  }
  return mean_binary_similarity;
}

void BtcDescManager::GenerateBtcDescs(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, const int frame_id,
    std::vector<BTC> &btcs_vec) {
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  init_voxel_map(input_cloud, voxel_map);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (config_setting_.print_debug_info_) {
    std::cout << "[BTC Gen] frame=" << frame_id 
              << ", planes=" << plane_cloud->size() << std::endl;
  }

  plane_cloud_vec_.push_back(plane_cloud);

  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);

  // P2-4: 增加调试统计信息 - 原始平面数量
  int original_plane_num = proj_plane_list.size();
  if (proj_plane_list.size() == 0) {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
        input_cloud->points[0].z;
    merge_plane_list.push_back(single_plane);
  } else {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }

  // P2-4: 增加调试统计信息 - 合并后平面数量和合并率
  int merged_plane_num = merge_plane_list.size();
  if (config_setting_.print_debug_info_) {
    float merge_ratio = (original_plane_num > 0) ?
        (1.0f - (float)merged_plane_num / (float)original_plane_num) : 0.0f;
    std::cout << "[BTC Gen] original_planes=" << original_plane_num
              << ", merged_planes=" << merged_plane_num
              << ", merge_ratio=" << merge_ratio * 100 << "%"
              << std::endl;
  }

  std::vector<BinaryDescriptor> binary_list;
  binary_extractor(merge_plane_list, input_cloud, binary_list);
  history_binary_list_.push_back(binary_list);
  if (config_setting_.print_debug_info_) {
    std::cout << "[BTC Gen] binary_descs=" << binary_list.size() << std::endl;
  }

  btcs_vec.clear();
  generate_btc(binary_list, frame_id, btcs_vec);
  if (config_setting_.print_debug_info_) {
    std::cout << "[BTC Gen] final_btcs=" << btcs_vec.size() << std::endl;
  }
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    delete (iter->second);
  }
  return;
}

void BtcDescManager::SearchLoop(
    const std::vector<BTC> &btcs_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<BTC, BTC>> &loop_std_pair,
    const Eigen::Vector3d &current_position) {
  if (btcs_vec.size() == 0) {
    std::cerr << "[BTC] No BTC descs!" << std::endl;
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BTCMatchList> candidate_matcher_vec;
  candidate_selector(btcs_vec, candidate_matcher_vec, current_position);

  // 收集候选帧ID用于诊断
  last_candidate_ids_.clear();
  for (auto& cm : candidate_matcher_vec) {
      last_candidate_ids_.push_back(cm.match_id_.second);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  double best_score = 0;
  int best_candidate_id = -1;
  int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<BTC, BTC>> best_sucess_match_vec;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<BTC, BTC>> sucess_match_vec;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec);
    if (config_setting_.print_debug_info_) {
      std::cout << "[Retrieval] try frame:"
                << candidate_matcher_vec[i].match_id_.second << ", rough size:"
                << candidate_matcher_vec[i].match_list_.size()
                << ", score:" << verify_score << std::endl;
    }

    if (verify_score > best_score) {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  if (config_setting_.print_debug_info_) {
    std::cout << "[Retrieval] best candidate:" << best_candidate_id
              << ", score:" << best_score 
              << " (threshold=" << config_setting_.icp_threshold_ << ")" << std::endl;
  }

  if (best_score > config_setting_.icp_threshold_) {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    
    if (config_setting_.print_debug_info_) {
      std::cout << "[BTC Loop] 检测到回环! frame=" << best_candidate_id 
                << "<->" << btcs_vec[0].frame_number_
                << ", score=" << best_score 
                << ", match_pairs=" << best_sucess_match_vec.size() << std::endl;
    }
    return;
  } else {
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
}

void BtcDescManager::AddBtcDescs(const std::vector<BTC> &btcs_vec,
                                  const Eigen::Vector3d &frame_position) {
  for (auto single_std : btcs_vec) {
    BTC_LOC position;
    // P1-1: 改用可配置量化，避免bucket太粗导致查询稳定性差
    // 使用std_side_resolution_作为hash分辨率（默认0.2m）
    double hash_resolution = config_setting_.std_side_resolution_;
    position.x = (int)(single_std.triangle_[0] / hash_resolution);
    position.y = (int)(single_std.triangle_[1] / hash_resolution);
    position.z = (int)(single_std.triangle_[2] / hash_resolution);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end()) {
      data_base_[position].push_back(single_std);
    } else {
      std::vector<BTC> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }

  // 新增：存储帧位置（用于搜索时预过滤）
  if (!btcs_vec.empty()) {
    int frame_id = btcs_vec[0].frame_number_;
    frame_positions_[frame_id] = frame_position;
  }
}

void BtcDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);
  ceres::LocalParameterization *quaternion_parameterization = new ceres::EigenQuaternionParameterization;
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr;
  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
  problem.AddParameterBlock(para_q, 4, quaternion_parameterization);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  int useful_match = 0;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
           normal_add.norm() < config_setting_.normal_threshold_) &&
          point_to_plane < config_setting_.dis_threshold_ &&
          point_to_point_dis < config_setting_.icp_point_to_point_max_) {
        useful_match++;
        ceres::CostFunction *cost_function;
        Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                   source_cloud->points[i].y,
                                   source_cloud->points[i].z);
        Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                    source_cloud->points[i].normal_y,
                                    source_cloud->points[i].normal_z);

        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
      }
    }
  }

  // P2-2: 增加ICP退化判断，有效匹配数太少则直接返回
  if (useful_match < config_setting_.icp_min_match_num_) {
    if (print_debug_info_) {
      std::cout << "[ICP] Useful match num: " << useful_match
                << " < min threshold: " << config_setting_.icp_min_match_num_
                << ", skip optimization to avoid degenerate case." << std::endl;
    }
    return;
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  transform.first = t;
  transform.second = rot;
}

void BtcDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->voxel_points_.push_back(p_c);
    } else {
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
  }
  #pragma omp parallel for
  for (size_t ii = 0; ii < index.size(); ii++) {
    iter_list[index[ii]]->second->init_octo_tree();
  }
}

void BtcDescManager::get_plane(
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      plane_cloud->push_back(pi);
    }
  }
}

void BtcDescManager::get_project_plane(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    std::vector<std::shared_ptr<Plane>> &project_plane_list) {
  std::vector<std::shared_ptr<Plane>> origin_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
  int current_id = 1;
  if (!origin_list.empty()) {
    for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
      for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
        Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
        Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
        double dis1 =
            fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
                 (*iter)->normal_(1) * (*iter2)->center_(1) +
                 (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
        double dis2 =
            fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
                 (*iter2)->normal_(1) * (*iter)->center_(1) +
                 (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
        if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
            normal_add.norm() < config_setting_.plane_merge_normal_thre_)
          if (dis1 < config_setting_.plane_merge_dis_thre_ &&
              dis2 < config_setting_.plane_merge_dis_thre_) {
            if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
              (*iter)->id_ = current_id;
              (*iter2)->id_ = current_id;
              current_id++;
            } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
              (*iter)->id_ = (*iter2)->id_;
            else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
              (*iter2)->id_ = (*iter)->id_;
          }
      }
    }
  }
  std::vector<std::shared_ptr<Plane>> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end()) continue;
    if (origin_list[i]->id_ == 0) {
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j) continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_++;
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        // 修复类型混用：Matrix3d应该用Matrix3d::Index或Eigen::Index
        Eigen::Matrix3d::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void BtcDescManager::merge_plane(
    std::vector<std::shared_ptr<Plane>> &origin_list,
    std::vector<std::shared_ptr<Plane>> &merge_plane_list) {
  if (origin_list.size() == 1) {
    merge_plane_list = origin_list;
    return;
  }

  // P0-3: 使用Union-Find进行真正的连通域合并
  // P1-3: 使用KDTree降低复杂度，只比较radius内的平面
  UnionFind uf(origin_list.size());

  // 构建KDTree，使用平面中心点
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane_centers(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < origin_list.size(); i++) {
    pcl::PointXYZ center;
    center.x = origin_list[i]->center_[0];
    center.y = origin_list[i]->center_[1];
    center.z = origin_list[i]->center_[2];
    plane_centers->push_back(center);
  }

  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(plane_centers);

  double search_radius = config_setting_.plane_merge_search_radius_;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;

  // 第一遍遍历：使用KDTree找到邻域平面，满足条件则Union
  for (size_t i = 0; i < origin_list.size(); i++) {
    pcl::PointXYZ searchPoint = plane_centers->points[i];
    if (kd_tree->radiusSearch(searchPoint, search_radius, pointIdxRadiusSearch,
                              pointRadiusSquaredDistance) > 0) {
      for (size_t idx = 0; idx < pointIdxRadiusSearch.size(); idx++) {
        size_t j = pointIdxRadiusSearch[idx];
        if (j <= i) continue;  // 避免重复比较

        Eigen::Vector3d normal_diff = origin_list[i]->normal_ - origin_list[j]->normal_;
        Eigen::Vector3d normal_add = origin_list[i]->normal_ + origin_list[j]->normal_;
        double dis1 =
            fabs(origin_list[i]->normal_(0) * origin_list[j]->center_(0) +
                 origin_list[i]->normal_(1) * origin_list[j]->center_(1) +
                 origin_list[i]->normal_(2) * origin_list[j]->center_(2) +
                 origin_list[i]->d_);
        double dis2 =
            fabs(origin_list[j]->normal_(0) * origin_list[i]->center_(0) +
                 origin_list[j]->normal_(1) * origin_list[i]->center_(1) +
                 origin_list[j]->normal_(2) * origin_list[i]->center_(2) +
                 origin_list[j]->d_);

        if ((normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
             normal_add.norm() < config_setting_.plane_merge_normal_thre_) &&
            dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_) {
          uf.unionSet(i, j);  // 合并满足条件的平面
        }
      }
    }
  }

  // 第二遍遍历：按root聚类
  std::unordered_map<int, std::vector<size_t>> clusters;
  for (size_t i = 0; i < origin_list.size(); i++) {
    int root = uf.find(i);
    clusters[root].push_back(i);
  }

  // 对每个聚类重新计算合并后的Plane
  merge_plane_list.clear();
  for (auto &cluster : clusters) {
    if (cluster.second.size() == 1) {
      // 单独的平面，不合并
      merge_plane_list.push_back(origin_list[cluster.second[0]]);
      origin_list[cluster.second[0]]->id_ = 0;
    } else {
      // 多个平面需要合并
      std::shared_ptr<Plane> merge_plane(new Plane);
      *merge_plane = *origin_list[cluster.second[0]];

      Eigen::Matrix3d P_PT_sum =
          (merge_plane->covariance_ +
           merge_plane->center_ * merge_plane->center_.transpose()) *
          merge_plane->points_size_;
      Eigen::Vector3d center_sum =
          merge_plane->center_ * merge_plane->points_size_;
      int total_points = merge_plane->points_size_;

      for (size_t idx = 1; idx < cluster.second.size(); idx++) {
        size_t j = cluster.second[idx];
        P_PT_sum +=
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        center_sum += origin_list[j]->center_ * origin_list[j]->points_size_;
        total_points += origin_list[j]->points_size_;
      }

      Eigen::Vector3d merge_center = center_sum / total_points;
      Eigen::Matrix3d merge_covariance =
          P_PT_sum / total_points - merge_center * merge_center.transpose();

      merge_plane->covariance_ = merge_covariance;
      merge_plane->center_ = merge_center;
      merge_plane->points_size_ = total_points;
      merge_plane->sub_plane_num_ = cluster.second.size();

      // 重新计算法向量和半径
      Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
      Eigen::Matrix3cd evecs = es.eigenvectors();
      Eigen::Vector3cd evals = es.eigenvalues();
      Eigen::Vector3d evalsReal = evals.real();
      Eigen::Matrix3d::Index evalsMin, evalsMax;
      evalsReal.rowwise().sum().minCoeff(&evalsMin);
      evalsReal.rowwise().sum().maxCoeff(&evalsMax);

      merge_plane->normal_ << evecs.real()(0, evalsMin),
          evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      merge_plane->min_eigen_value_ = evalsReal(evalsMin);
      merge_plane->radius_ = sqrt(evalsReal(evalsMax));
      merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                         merge_plane->normal_(1) * merge_plane->center_(1) +
                         merge_plane->normal_(2) * merge_plane->center_(2));

      merge_plane->p_center_.x = merge_plane->center_(0);
      merge_plane->p_center_.y = merge_plane->center_(1);
      merge_plane->p_center_.z = merge_plane->center_(2);
      merge_plane->p_center_.normal_x = merge_plane->normal_(0);
      merge_plane->p_center_.normal_y = merge_plane->normal_(1);
      merge_plane->p_center_.normal_z = merge_plane->normal_(2);

      // 设置合并后的Plane ID
      merge_plane->id_ = cluster.first + 1;  // 使用root+1作为新ID
      merge_plane_list.push_back(merge_plane);
    }
  }

  return;
}

void BtcDescManager::binary_extractor(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list) {
  binary_descriptor_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    int plane_id = proj_plane_list[i]->id_;  // P2-1: 使用Plane的ID
    if (proj_normal.z() < 0) {
      proj_normal = -proj_normal;
    }
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      if (print_debug_info_) {
        std::cout << "[Description] reference plane normal:"
                  << proj_normal.transpose()
                  << ", center:" << proj_center.transpose() << std::endl;
      }
      useful_proj_num++;
      extract_binary(proj_center, proj_normal, input_cloud,
                     prepare_binary_list, plane_id);  // P0-1: 传入plane_id
      for (auto bi : prepare_binary_list) {
        temp_binary_list.push_back(bi);
      }
      if (useful_proj_num == config_setting_.proj_plane_num_) {
        break;
      }
    }
  }
  non_maxi_suppression(temp_binary_list);
  if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
  } else {
    std::sort(temp_binary_list.begin(), temp_binary_list.end(),
              binary_greater_sort);
    for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[i]);
    }
  }
  return;
}

void BtcDescManager::extract_binary(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list,
    int plane_id) {
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;

  // P0-2: 使用Gram-Schmidt方法生成稳定的局部坐标系，避免数值退化
  Eigen::Vector3d ref;
  if (fabs(project_normal.z()) < 0.9) {
    ref = Eigen::Vector3d(0, 0, 1);  // 当法向不接近Z轴时，用Z轴作为参考
  } else {
    ref = Eigen::Vector3d(1, 0, 0);  // 当法向接近Z轴时，用X轴作为参考
  }
  Eigen::Vector3d x_axis = project_normal.cross(ref).normalized();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis).normalized();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  // 修复内存泄漏：使用std::vector替代原始指针的二维数组
  // 即使有提前返回或异常，STL也会自动释放内存
  std::vector<std::vector<std::vector<double>>> dis_container(
      x_axis_len, std::vector<std::vector<double>>(y_axis_len));
  std::vector<std::vector<BinaryDescriptor>> binary_container(
      x_axis_len, std::vector<BinaryDescriptor>(y_axis_len));
  std::vector<std::vector<double>> img_count(
      x_axis_len, std::vector<double>(y_axis_len, 0.0));
  std::vector<std::vector<double>> dis_array(
      x_axis_len, std::vector<double>(y_axis_len, 0.0));
  std::vector<std::vector<double>> mean_x_list(
      x_axis_len, std::vector<double>(y_axis_len, 0.0));
  std::vector<std::vector<double>> mean_y_list(
      x_axis_len, std::vector<double>(y_axis_len, 0.0));

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        single_binary.normal_ = project_normal;  // P0-1: 保存投影平面的法向量
        single_binary.plane_id_ = plane_id;  // P2-1: 保存投影平面的ID
        binary_container[x][y] = single_binary;
      }
    }
  }

  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis) {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      continue;
    }
    bool is_add = true;

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  // 修复内存泄漏：使用std::vector后，无需手动delete，STL自动管理内存
}

void BtcDescManager::non_maxi_suppression(
    std::vector<BinaryDescriptor> &binary_list) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary_);
    is_add_list.push_back(true);
  }
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting_.non_max_suppression_radius_;
  double score_margin = config_setting_.nms_score_margin_;  // P1-4: NMS score margin

  for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
    pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0) {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i) {
          continue;
        }
        // P1-4: 只有邻居score > 自己score + margin时才删除，避免大量误删
        if (pre_count_list[pointIdxRadiusSearch[j]] >
            pre_count_list[i] + score_margin) {
          is_add_list[i] = false;
          break;  // 已经确定要删除，不需要继续检查其他邻居
        }
      }
    }
  }
  std::vector<BinaryDescriptor> pass_binary_list;
  for (size_t i = 0; i < is_add_list.size(); i++) {
    if (is_add_list[i]) {
      pass_binary_list.push_back(binary_list[i]);
    }
  }
  binary_list.clear();
  for (auto var : pass_binary_list) {
    binary_list.push_back(var);
  }
  return;
}

void BtcDescManager::generate_btc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<BTC> &btc_list) {
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = config_setting_.descriptor_near_num_;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < K - 1; m++) {
        for (int n = m + 1; n < K; n++) {
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxNKNSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxNKNSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxNKNSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxNKNSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxNKNSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxNKNSearch[n]];
            }
            BTC single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            // P0-1: 修复法向量未初始化问题，从BinaryDescriptor中获取法向量
            normal_1 = binary_A.normal_;
            normal_2 = binary_B.normal_;
            normal_3 = binary_C.normal_;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.frame_number_ = frame_id;
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            feat_map[position] = true;
            btc_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void BtcDescManager::candidate_selector(
    const std::vector<BTC> &current_STD_list,
    std::vector<BTCMatchList> &candidate_matcher_vec,
    const Eigen::Vector3d &current_position) {
  int current_frame_id = current_STD_list[0].frame_number_;
  int outlier = 0;
  double max_dis = 50;

  // 统计变量（原子类型，用于并行执行）
  std::atomic<int> total_hash_hits{0};
  std::atomic<int> total_skip_near{0};
  std::atomic<int> total_dis_filter{0};
  std::atomic<int> total_sim_filter{0};

  // 修复数组越界：改用unordered_map，避免硬编码20000帧上限
  std::unordered_map<int, int> match_votes;  // frame_number -> vote count
  std::vector<std::pair<BTC, BTC>> match_list;
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<BTC_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;

  int query_num = 0;
  int pass_num = 0;
  // P1-1: 使用可配置hash分辨率
  double hash_resolution = config_setting_.std_side_resolution_;
  #pragma omp parallel for
  for (size_t ii = 0; ii < index.size(); ii++) {
    const size_t &i = index[ii];
    BTC descriptor = current_STD_list[i];
        BTC_LOC position;
        int best_index = 0;
        BTC_LOC best_position;
        double dis_threshold =
            descriptor.triangle_.norm() *
            config_setting_.rough_dis_threshold_;
        for (auto voxel_inc : voxel_round) {
          // P1-1: 改用可配置量化，与AddBtcDescs保持一致
          position.x = (int)(descriptor.triangle_[0] / hash_resolution) + voxel_inc[0];
          position.y = (int)(descriptor.triangle_[1] / hash_resolution) + voxel_inc[1];
          position.z = (int)(descriptor.triangle_[2] / hash_resolution) + voxel_inc[2];
          Eigen::Vector3d voxel_center((double)position.x * hash_resolution + hash_resolution / 2,
                                       (double)position.y * hash_resolution + hash_resolution / 2,
                                       (double)position.z * hash_resolution + hash_resolution / 2);
          if ((descriptor.triangle_ - voxel_center).norm() < 1.5 * hash_resolution) {
            auto iter = data_base_.find(position);
            if (iter != data_base_.end()) {
              total_hash_hits++;  // 统计：哈希表命中次数
              bool is_push_position = false;
              for (size_t j = 0; j < data_base_[position].size(); j++) {
                int candidate_frame_id = data_base_[position][j].frame_number_;

                // 预过滤1: 跳过邻近帧（帧号差值）
                if ((descriptor.frame_number_ - candidate_frame_id) <=
                    config_setting_.skip_near_num_) {
                  total_skip_near++;  // 统计：跳过邻近帧
                  continue;
                }

                // 预过滤2: 根据odom距离过滤（新增）
                if (!frame_positions_.empty() && current_position.norm() > 0) {
                  auto curr_pos_iter = frame_positions_.find(descriptor.frame_number_);
                  auto cand_pos_iter = frame_positions_.find(candidate_frame_id);
                  if (curr_pos_iter != frame_positions_.end() &&
                      cand_pos_iter != frame_positions_.end()) {
                    double odom_distance = (curr_pos_iter->second - cand_pos_iter->second).norm();
                    if (odom_distance > max_loop_distance_) {
                      continue;  // 跳过距离太远的候选帧
                    }
                  }
                }

                // BTC几何匹配
                double dis =
                    (descriptor.triangle_ - data_base_[position][j].triangle_)
                        .norm();
                if (dis < dis_threshold) {
                    double similarity =
                        (binary_similarity(descriptor.binary_A_,
                                           data_base_[position][j].binary_A_) +
                         binary_similarity(descriptor.binary_B_,
                                           data_base_[position][j].binary_B_) +
                         binary_similarity(descriptor.binary_C_,
                                           data_base_[position][j].binary_C_)) /
                        3;
                    if (similarity > config_setting_.similarity_threshold_) {
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                    } else {
                      total_sim_filter++;  // 统计：相似度过滤
                    }
                  } else {
                    total_dis_filter++;  // 统计：距离过滤
                  }
              }
            }
          }
      }
  }

  // 打印统计信息（如果启用debug）
  if (config_setting_.print_debug_info_) {
    int num_useful = 0;
    for (size_t i = 0; i < useful_match.size(); i++) {
      if (useful_match[i]) num_useful++;
    }
    std::cout << "[candidate_selector] frame=" << current_frame_id
              << ", btcs=" << current_STD_list.size()
              << ", hash_hits=" << total_hash_hits
              << ", skip_near=" << total_skip_near
              << ", dis_filter=" << total_dis_filter
              << ", sim_filter=" << total_sim_filter
              << ", useful_match=" << num_useful << "/" << current_STD_list.size()
              << std::endl;
  }
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {
        // 修复数组越界：使用unordered_map累加投票
        match_votes[data_base_[useful_match_position[i][j]]
                        [useful_match_index[i][j]]
                            .frame_number_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        match_list_index.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_number_);
      }
    }
  }
  bool multi_thread_en = false;
  if (multi_thread_en) {
    #pragma omp parallel for
    for (size_t ii = 0; ii < index.size(); ii++) {
      const size_t &i = index[ii];
      if (useful_match[i]) {
        std::pair<BTC, BTC> single_match_pair;
        single_match_pair.first = current_STD_list[i];
        for (size_t j = 0; j < useful_match_index[i].size(); j++) {
          single_match_pair.second = data_base_[useful_match_position[i][j]]
                                               [useful_match_index[i][j]];
          mylock.lock();
          // 修复数组越界：使用unordered_map累加投票
          match_votes[single_match_pair.second.frame_number_]++;
          match_list.push_back(single_match_pair);
          match_list_index.push_back(
              single_match_pair.second.frame_number_);
          mylock.unlock();
        }
      }
    }
  }

  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    // 修复数组越界：遍历unordered_map找最大投票
    for (const auto& vote_pair : match_votes) {
      if (vote_pair.second > max_vote) {
        max_vote = vote_pair.second;
        max_vote_index = vote_pair.first;
      }
    }
    BTCMatchList match_triangle_list;
    if (max_vote_index >= 0 && max_vote >= config_setting_.candidate_selector_min_vote_) {
      match_votes[max_vote_index] = 0;  // 清零已选中的候选
      match_triangle_list.match_frame_ = max_vote_index;
      match_triangle_list.match_id_.first = current_frame_id;
      match_triangle_list.match_id_.second = max_vote_index;
      double mean_dis = 0;
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          std::pair<BTC, BTC> single_match_pair;
          single_match_pair.first = current_STD_list[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    }
  }
}

void BtcDescManager::candidate_verify(
    const BTCMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<BTC, BTC>> &sucess_match_list) {
  sucess_match_list.clear();

  // P2-3: 增加几何一致性过滤，减少ICP误收敛
  // 计算三角形边长方差和中心分布
  if (candidate_matcher.match_list_.size() < (size_t)config_setting_.candidate_verify_min_pairs_) {
    verify_score = -1;
    return;
  }

  // 计算triangle side的均值和方差
  double side_mean = 0;
  double side_var = 0;
  for (const auto &match : candidate_matcher.match_list_) {
    side_mean += match.first.triangle_.norm();
  }
  side_mean /= candidate_matcher.match_list_.size();

  for (const auto &match : candidate_matcher.match_list_) {
    double diff = match.first.triangle_.norm() - side_mean;
    side_var += diff * diff;
  }
  side_var /= candidate_matcher.match_list_.size();
  double side_std = sqrt(side_var);

  // 计算center分布的均值和方差
  Eigen::Vector3d center_mean(0, 0, 0);
  double center_var = 0;
  for (const auto &match : candidate_matcher.match_list_) {
    center_mean += match.first.center_;
  }
  center_mean /= candidate_matcher.match_list_.size();

  for (const auto &match : candidate_matcher.match_list_) {
    double diff = (match.first.center_ - center_mean).norm();
    center_var += diff * diff;
  }
  center_var /= candidate_matcher.match_list_.size();
  double center_std = sqrt(center_var);

  // 几何一致性阈值（从配置文件读取）
  double max_side_std_threshold = config_setting_.geom_side_std_threshold_;
  double max_center_std_threshold = config_setting_.geom_center_std_threshold_;

  if (side_std > max_side_std_threshold || center_std > max_center_std_threshold) {
    if (config_setting_.print_debug_info_) {
      std::cout << "[Verify] Geometric consistency check failed: "
                << "side_std=" << side_std << " (threshold=" << max_side_std_threshold << "), "
                << "center_std=" << center_std << " (threshold=" << max_center_std_threshold << ")"
                << std::endl;
    }
    verify_score = -1;
    return;
  }

  double dis_threshold = config_setting_.ransac_correspondence_dis_;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  int skip_len = (int)(candidate_matcher.match_list_.size() / config_setting_.ransac_sample_max_) + 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;
  #pragma omp parallel for
  for (size_t ii = 0; ii < index.size(); ii++) {
    const size_t &i = index[ii];
    auto single_pair = candidate_matcher.match_list_[i * skip_len];
        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_t, test_rot);
        for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
          auto verify_pair = candidate_matcher.match_list_[j];
          Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
          Eigen::Vector3d A_transform = test_rot * A + test_t;
          Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
          Eigen::Vector3d B_transform = test_rot * B + test_t;
          Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
          Eigen::Vector3d C_transform = test_rot * C + test_t;
          double dis_A =
              (A_transform - verify_pair.second.binary_A_.location_).norm();
          double dis_B =
              (B_transform - verify_pair.second.binary_B_.location_).norm();
          double dis_C =
              (C_transform - verify_pair.second.binary_C_.location_).norm();
          if (dis_A < dis_threshold && dis_B < dis_threshold &&
              dis_C < dis_threshold) {
            vote++;
          }
        }
        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
  }

  int max_vote_index = 0;
  int max_vote = 0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (max_vote < vote_list[i]) {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }
  
  if (config_setting_.print_debug_info_) {
    std::cout << "[Verify] RANSAC max_vote=" << max_vote 
              << "/" << candidate_matcher.match_list_.size()
              << " (threshold=" << config_setting_.ransac_min_vote_ << ")" << std::endl;
  }
  
  if (max_vote >= config_setting_.ransac_min_vote_) {
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    // 初始位姿：单对求解
    triangle_solver(best_pair, best_t, best_rot);

    if (config_setting_.print_debug_info_) {
      std::cout << "[Verify] RANSAC init: t=" << best_t.norm() << "m" << std::endl;
    }

    // 迭代RANSAC: 用当前位姿找inlier → 用inlier重算位姿 → 反复直到收敛
    int prev_inlier_count = 0;
    const int max_iterations = config_setting_.ransac_max_iterations_;
    for (int iter = 0; iter < max_iterations; iter++) {
      std::vector<std::pair<BTC, BTC>> inlier_pairs;
      for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
        auto verify_pair = candidate_matcher.match_list_[j];
        Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
        Eigen::Vector3d A_transform = best_rot * A + best_t;
        Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
        Eigen::Vector3d B_transform = best_rot * B + best_t;
        Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
        Eigen::Vector3d C_transform = best_rot * C + best_t;
        double dis_A =
            (A_transform - verify_pair.second.binary_A_.location_).norm();
        double dis_B =
            (B_transform - verify_pair.second.binary_B_.location_).norm();
        double dis_C =
            (C_transform - verify_pair.second.binary_C_.location_).norm();
        if (dis_A < dis_threshold && dis_B < dis_threshold &&
            dis_C < dis_threshold) {
          inlier_pairs.push_back(verify_pair);
        }
      }

      if (config_setting_.print_debug_info_) {
        std::cout << "[Verify] RANSAC iter " << iter << ": inliers=" << inlier_pairs.size() << std::endl;
      }

      // 收敛判断
      if ((int)inlier_pairs.size() == prev_inlier_count) {
        if (config_setting_.print_debug_info_) {
          std::cout << "[Verify] RANSAC converged at " << inlier_pairs.size() << " inliers" << std::endl;
        }
        break;
      }
      prev_inlier_count = inlier_pairs.size();

      // 用inlier重算位姿（多对SVD优化）
      if (inlier_pairs.size() > 1) {
        triangle_solver_multi(inlier_pairs, best_t, best_rot);
        if (config_setting_.print_debug_info_) {
          std::cout << "[Verify] RANSAC iter " << iter << ": refined t=" << best_t.norm() << "m" << std::endl;
        }
      }
      sucess_match_list = inlier_pairs;
    }

    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    verify_score = plane_geometric_verify(
        plane_cloud_vec_.back(),
        plane_cloud_vec_[candidate_matcher.match_id_.second], relative_pose);
    
    if (config_setting_.print_debug_info_) {
      std::cout << "[Verify] RANSAC success! match_pairs=" << sucess_match_list.size()
                << ", final_t=" << best_t.norm() << "m"
                << ", plane_verify_score=" << verify_score << std::endl;
    }
  } else {
    verify_score = -1;
    if (config_setting_.print_debug_info_) {
      std::cout << "[Verify] RANSAC failed (max_vote=" << max_vote << " < " << config_setting_.ransac_min_vote_ << ")" << std::endl;
    }
  }
  return;
}

void BtcDescManager::triangle_solver(std::pair<BTC, BTC> &std_pair,
                                     Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
  src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
  src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

void BtcDescManager::triangle_solver_multi(
    const std::vector<std::pair<BTC, BTC>> &inlier_pairs,
    Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  if (inlier_pairs.empty()) return;

  // 收集所有顶点对 (src -> ref)
  // 每个BTC对提供3个顶点：A, B, C
  // 减去各自重心做SVD
  Eigen::MatrixXd src(3, inlier_pairs.size() * 3);
  Eigen::MatrixXd ref(3, inlier_pairs.size() * 3);

  int col = 0;
  Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_centroid = Eigen::Vector3d::Zero();
  int total_pts = 0;
  for (const auto &pair : inlier_pairs) {
    src_centroid += pair.first.binary_A_.location_;
    src_centroid += pair.first.binary_B_.location_;
    src_centroid += pair.first.binary_C_.location_;
    ref_centroid += pair.second.binary_A_.location_;
    ref_centroid += pair.second.binary_B_.location_;
    ref_centroid += pair.second.binary_C_.location_;
    total_pts += 3;
  }
  src_centroid /= total_pts;
  ref_centroid /= total_pts;

  col = 0;
  for (const auto &pair : inlier_pairs) {
    src.col(col) = pair.first.binary_A_.location_ - src_centroid;
    ref.col(col) = pair.second.binary_A_.location_ - ref_centroid;
    col++;
    src.col(col) = pair.first.binary_B_.location_ - src_centroid;
    ref.col(col) = pair.second.binary_B_.location_ - ref_centroid;
    col++;
    src.col(col) = pair.first.binary_C_.location_ - src_centroid;
    ref.col(col) = pair.second.binary_C_.location_ - ref_centroid;
    col++;
  }

  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * src_centroid + ref_centroid;
}

double BtcDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
  Eigen::Vector3d t = transform.first;
  Eigen::Matrix3d rot = transform.second;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = config_setting_.normal_threshold_;
  double dis_threshold = config_setting_.dis_threshold_;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}