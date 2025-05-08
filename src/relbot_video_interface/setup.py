from setuptools import setup

package_name = 'relbot_video_interface'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch', ['launch/video_interface.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Bavantha Udugama',
    maintainer_email='b.udugama@utwente.nl',
    description='Capture GStreamer video and publish object positions to RELBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_interface = relbot_video_interface.video_interface_node:main'
        ],
    },
)