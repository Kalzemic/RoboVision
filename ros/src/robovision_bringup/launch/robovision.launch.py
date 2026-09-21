from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

 
    robovision_trt = ComposableNode(
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        name='robovision_trt',
        parameters=[{
            'engine_file_path':os.path.join(get_package_share_directory('robovision_bringup'),'models','RoboVision.trt'),
            'input_tensor_names':['images'],
            'input_binding_names':['images'],
            'output_tensor_names': ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',],
            'output_binding_names': [ 'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',],
            'force_engine_update': False,

        }],
        remappings=[
            ('tensor_pub', 'input'),
            ('tensor_sub', 'output'),
        ]
    )

    encoder = ComposableNode(
        package='robovision_bringup',
        plugin='robovision::Encoder',
         name='encoder',
         remappings=[
            ('image_sub', 'image'),
            ('tensor_pub', 'input'),
        ]
    )

    object_selector = ComposableNode(
        package='robovision_bringup',
        plugin='robovision::ObjectSelector',
        name='object_selector',
        remappings=[
            ('tensor_sub','output'),
            ('tensor_pub','input')
        ]
    )
    container = ComposableNodeContainer(
        name='robovision_container',
        namespace='robovision',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[encoder, robovision_trt, object_selector],
        output='screen'
    )

    return LaunchDescription([container])