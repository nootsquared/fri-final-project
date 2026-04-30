## Setup

Use Ctr+Shift+V to paste in terminator.

```
# you shouldn't need to run this one (packages should already be installed)
sudo apt update && sudo apt install libgtk-3-dev libapriltag-dev
```
```
pip install --upgrade packaging --user
```
```
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'export COLCON_WS=~/bwi_ros2' >> ~/.bashrc 
mkdir bwi_ros2
cd bwi_ros2
mkdir src
```
```
cd ~/bwi_ros2
git clone https://github.com/UT-Austin-FRI-Autonomous-Robots/serial.git
```
```
cd ~/bwi_ros2/serial/
rm -rf build
mkdir build
cd build
cmake ..
make
```
```
cd ~/bwi_ros2/src
git clone --branch humble https://github.com/microsoft/Azure_Kinect_ROS_Driver.git
# Humble: upstream still uses rclcpp::Duration(0.25), which does not compile.
# From your fri-final-project clone, run (adjust FRI_REPO to where you cloned it):
#   cd ~/bwi_ros2/src/Azure_Kinect_ROS_Driver
#   patch -p1 < $FRI_REPO/ros_code/patches/azure_kinect_ros_driver_humble_duration.patch
git clone https://github.com/Living-With-Robots-Lab/apriltag_ros.git
git clone https://github.com/utexas-bwi/bwi_ros2_common.git
git clone https://github.com/Living-With-Robots-Lab/segbot_description.git
git clone --recurse-submodules https://github.com/utexas-bwi/urg_node2.git
git clone https://github.com/Living-With-Robots-Lab/lidar.git
```

# On a V2
```
cd ~/bwi_ros2/src
git clone https://github.com/utexas-bwi/libsegwayrmp_ros2.git
mv libsegwayrmp_ros2/ libsegwayrmp
git clone https://github.com/utexas-bwi/segway_rmp_ros2.git
```

# Build
```
source ~/.bashrc
cd ~/bwi_ros2
source /opt/ros/humble/setup.bash
rosdep update
# segbot_description lists pr2_description; rosdep may have no apt rule. Skip or install:
#   sudo apt install ros-humble-pr2-description   # if available on your distro
rosdep install --from-paths src -y --ignore-src --skip-keys pr2_description
colcon build
source install/setup.bash
```
There will be warnings after you build for the first time, but hopefully no errors. Remember to run colcon build in your workspace root everytime you make a change.
