cmake_minimum_required(VERSION 3.5)
project(sc_pgo_ros2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# Python binding option (默认开启，确保离线脚本可直接调用C++ BTC)
option(BUILD_PYTHON_BINDING "Build Python binding for BTC/SC" ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
# 只请求必要PCL组件（避免引入OpenCV → TBB依赖）
find_package(PCL REQUIRED COMPONENTS common filters search kdtree octree registration io)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# ndt_omp_ros2 dependency for GICP_OMP
find_package(ndt_omp_ros2 REQUIRED)

# Python binding dependencies (可选)
if(BUILD_PYTHON_BINDING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)
  message(STATUS "Python binding enabled - Python3: ${Python3_VERSION}, pybind11: ${pybind11_VERSION}")
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# ==================== 主可执行文件 (ROS2节点) ====================
add_executable(alaserPGO 
  src/laserPosegraphOptimization.cpp
  src/btc.cpp
  src/gicp_registration.cpp
  src/dynamic_remove.cpp
)
ament_target_dependencies(alaserPGO
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs visualization_msgs tf2 tf2_ros
  cv_bridge pcl_conversions pcl_msgs ament_index_cpp ndt_omp_ros2 std_srvs
)
target_compile_options(alaserPGO PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(alaserPGO
  ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  OpenMP::OpenMP_CXX
  gtsam
)

install(TARGETS
  alaserPGO
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

install(
  DIRECTORY rviz_cfg config
  DESTINATION share/${PROJECT_NAME}
)

# ==================== Python Binding (可选) ====================
if(BUILD_PYTHON_BINDING)
  # 精简PCL库列表（不包含visualization/io等引入OpenCV→TBB的组件）
  set(PCL_CORE_LIBS pcl_common pcl_filters pcl_search pcl_kdtree)

  # TBB: Ceres或PCL的间接依赖（系统库预编译时已引入）
  find_package(TBB QUIET)
  if(TBB_FOUND)
    set(TBB_LIBS TBB::tbb)
  else()
    set(TBB_LIBS tbb)  # fallback to raw linker flag
  endif()

  # BTC Python binding - PCL核心 + Ceres + OpenCV(BTC用它读YAML配置) + TBB(间接)
    pybind11_add_module(btc_cpp
      src/btc_python_binding.cpp
      src/btc.cpp
    )
    target_link_libraries(btc_cpp PRIVATE
      ${PCL_CORE_LIBS}
      ${CERES_LIBRARIES}
      ${OpenCV_LIBS}
      ${TBB_LIBS}
    )
    target_include_directories(btc_cpp PRIVATE
      include
      ${PCL_INCLUDE_DIRS}
      ${CERES_INCLUDE_DIRS}
      ${OpenCV_INCLUDE_DIRS}
    )

  # ScanContext Python binding - 只依赖PCL核心组件
  pybind11_add_module(sc_cpp
    src/sc_python_binding.cpp
    include/scancontext/Scancontext.cpp
  )
  target_link_libraries(sc_cpp PRIVATE ${PCL_CORE_LIBS})
  target_include_directories(sc_cpp PRIVATE
    include
    ${PCL_INCLUDE_DIRS}
  )

  # 安装Python模块
  install(TARGETS btc_cpp sc_cpp
    LIBRARY DESTINATION lib/python3/dist-packages
  )
  message(STATUS "Python bindings (btc_cpp, sc_cpp) will be installed to: lib/python3/dist-packages")

  # GICP_OMP Python binding - 只依赖PCL核心组件 + OpenMP
  set(_ndt_omp_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ndt_omp_ros2/src/pclomp)
  pybind11_add_module(gicp_omp_cpp
    src/gicp_python_binding.cpp
    ${_ndt_omp_src_dir}/gicp_omp.cpp
    ${_ndt_omp_src_dir}/voxel_grid_covariance_omp.cpp
  )
  target_link_libraries(gicp_omp_cpp PRIVATE
    ${PCL_CORE_LIBS}
  )
  target_include_directories(gicp_omp_cpp PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ndt_omp_ros2/include
  )
  target_compile_options(gicp_omp_cpp PRIVATE ${OpenMP_CXX_FLAGS})

  install(TARGETS gicp_omp_cpp
    LIBRARY DESTINATION lib/python3/dist-packages
  )
  message(STATUS "Python binding 'gicp_omp_cpp' will be installed to: lib/python3/dist-packages")
endif()

ament_package()
