from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='relbot_video_interface',
            executable='video_interface',
            name='video_interface',
            output='screen',
            parameters=[
                {'gst_pipeline': (
                    'udpsrc port=5000 caps="application/x-rtp,media=video,'
                    'encoding-name=H264,payload=96" ! '
                    'rtph264depay ! avdec_h264 ! videoconvert ! '
                    'video/x-raw,format=RGB ! appsink name=sink'
                )}
            ],
        ),
    ])