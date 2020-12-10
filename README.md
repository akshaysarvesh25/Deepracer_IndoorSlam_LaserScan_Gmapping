# Deepracer_IndoorSlam_LaserScan_Gmapping
Indoor 2D SLAM of deepracer without IMU using Laser Scan matcher and Gmapping. This repository uses only Lidar data (on /scan topic) to perform SLAM.

## To get started : 
1. To launch the SLAM on an indoor bagfile data : ```roslaunch laser_scan_matcher deepracer_gmapping_slam.launch```<br/>
2. Edit and add new bagfiles in the /demo/ directory ```roscd laser_scan_matcher/demo/ ```<br/>
3. Edit the name of the bagfile in the deepracer_gmapping_slam.launch file.

## Useful links :
Laser Scan matcher link : https://github.com/ccny-ros-pkg/scan_tools<br/>
Gmapping link : https://github.com/ros-perception/slam_gmapping <br/>
Control Deepracer with a joystick : https://github.com/athackst/deepracer_joy
