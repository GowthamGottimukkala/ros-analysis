# ROS File Analysis
One fundamental pillar of Autonomous Vehicles (AVs) is perception, which uses sensors such as cameras and LiDARs (Light Detection and Ranging) to understand the driving environment. Multiple previous efforts have been made to study the security of perception systems due to its direct impact on road safety. In this project, we aim to build a system that can see discrepancies in the specific communication channels that the attack aimed. It studies, analyzes, and quantize these variations. Using a pipeline that fetches detected objects from the LIDAR sensor and the object detection model in the Autoware system built on the ROS middleware, our system can produce results on these variations.

## Scripts

### bagtopointcloud-filter.py
- This program takes in a bag file as input and publishes the point cloud messages. Additionally, it can also take min and max values in x,y,z dimensions to filter the point clouds before publishing the point cloud messages. If we use pcl_ros pointcloud_to_pcd with this program, we can save the filtered point cloud messages in pcd files
- Installaton
  - tba
- How To Run
  - `python3 bagtopointcloud-filter.py -b bagpath.bag -topic /inputtopic -xmin 0 -xmax 50 -ymin 0 -ymax 50 -zmin 0 -zmax 50 -tmin 0 -tmax 50 -pub /outputtopic`
  
### pcdtopointcloud.py
- This program takes in a folder of pcd files as input, reads them and publishes the point cloud messages. If we use rosbag record with this, we can convert pcd files to an ROS bag
- Installation
  - tba
- How To Run
  - `python3 pcdtopointcloud.py -p pcdfilespath/ -pub outputtopic/`
  
### bintopcd.py
- This program takes in point cloud message in .bin format and convert it to .pcd format
- Installation
  - tba
- How To Run
  - `python3 bintopcd.py`
