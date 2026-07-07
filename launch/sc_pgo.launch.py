#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rvizscpgo_arg = DeclareLaunchArgument(
        "rvizscpgo", default_value="true", description="Launch RViz for SC-PGO"
    )

    pkg_share_dir = get_package_share_directory("sc_pgo_ros2")
    config_file = PathJoinSubstitution([pkg_share_dir, "config", "btc_config.yaml"])

    alaserPGO_node = Node(
        package="sc_pgo_ros2",
        executable="alaserPGO",
        name="alaserPGO",
        output="screen",
        parameters=[config_file],
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

    return LaunchDescription([rvizscpgo_arg, alaserPGO_node])
