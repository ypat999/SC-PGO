#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    rvizscpgo_arg = DeclareLaunchArgument(
        "rvizscpgo", default_value="true", description="Launch RViz for SC-PGO"
    )
    
    save_directory_arg = DeclareLaunchArgument(
        "save_directory", 
        default_value="/home/ywj/save_data/", 
        description="Directory to save map PCD, optimized poses and keyframe data"
    )
    
    save_map_service_name_arg = DeclareLaunchArgument(
        "save_map_service_name", 
        default_value="save_map", 
        description="Service name for saving map"
    )
    
    map_filename_arg = DeclareLaunchArgument(
        "map_filename", 
        default_value="map.pcd", 
        description="Filename of the saved map PCD file"
    )

    alaserPGO_node = Node(
        package="sc_pgo_ros2",
        executable="alaserPGO",
        name="alaserPGO",
        output="screen",
        parameters=[
            {"scan_line": 4},
            {"minimum_range": 0.3},
            {"mapping_line_resolution": 0.4},
            {"mapping_plane_resolution": 0.8},
            {"mapviz_filter_size": 0.05},
            {"keyframe_meter_gap": 5.0},
            {"sc_dist_thres": 0.3},
            {"sc_max_radius": 290.0},
            {"save_directory": LaunchConfiguration("save_directory")},
            {"save_map_service_name": LaunchConfiguration("save_map_service_name")},
            {"map_filename": LaunchConfiguration("map_filename")},
            # GICP parameters for loop closure refinement
            {"use_gicp_for_loop_closure": True},
            {"gicp_fitness_score_threshold": 0.5},
            {"gicp_transformation_epsilon": 1e-6},
            {"gicp_max_correspondence_distance": 30.0},
            {"gicp_max_iterations": 100},
            {"gicp_num_threads": 4},
        ],
        remappings=[
            # ("/aft_mapped_to_init", "/Odometry"),
            # ("/aft_mapped_to_init", "/lio/odom"),
            # ("/velodyne_cloud_registered_local", "/lio/body/cloud"),
            # ("/cloud_for_scancontext", "/lio/cloud_world"),
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
        prefix=['taskset -c 6'],   # 绑定 CPU 4
    )

    # RViz Node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rvizscpgo",
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rvizscpgo")),
    )

    return LaunchDescription([rvizscpgo_arg, save_directory_arg, save_map_service_name_arg, map_filename_arg, alaserPGO_node])
