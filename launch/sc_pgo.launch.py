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

    pkg_share_dir = get_package_share_directory("sc_pgo_ros2")
    config_file = PathJoinSubstitution([pkg_share_dir, "config", "btc_config.yaml"])

    alaserPGO_node = Node(
        package="sc_pgo_ros2",
        executable="alaserPGO",
        name="alaserPGO",
        output="screen",
        parameters=[
            config_file,
            {"save_directory": LaunchConfiguration("save_directory")},
            {"save_map_service_name": LaunchConfiguration("save_map_service_name")},
            {"map_filename": LaunchConfiguration("map_filename")},
        ],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
        prefix=['taskset -c 6'],
    )

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
