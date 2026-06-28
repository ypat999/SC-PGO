#pragma once
// ScanContext 计时工具最小化stub（替代ROS版本的aloam_velodyne/tic_toc.h）
// 所有计时操作均为空操作

#include <string>

class TicTocV2 {
public:
    TicTocV2() {}
    void tic() {}                // 开始计时（空操作）
    double toc(const std::string& = "") { return 0.0; }  // 结束计时（空操作）
};
