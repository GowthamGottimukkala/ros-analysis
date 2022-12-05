from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import ImageFile
from ros import rosbag
from open3d import *
from pypcd import pypcd
from os import listdir
from os.path import isfile, join
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
import rospy, time, argparse, std_msgs.msg, struct, cv2
import sensor_msgs.point_cloud2 as pcl2

device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32


# publishes point cloud messages given pcd files folder on given topic
# Need pypcd python package. Can be installed using `python -m pip install --user git+https://github.com/DanielPollithy/pypcd.git`
def publish_pointcloud_msgs(folderPath, publishingTopic):
    pcd_files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    pub = rospy.Publisher(publishingTopic, PointCloud2)
    rospy.init_node('pointcloudpublisher', anonymous=True)
    for fileName in pcd_files:
        pointCloudMsg = convert_pcd_to_pointcloud_msg(folderPath + fileName)
        pub.publish(pointCloudMsg)
def convert_pcd_to_pointcloud_msg(pcdFilePath):
    fields = [PointField('x', 0, PointField.FLOAT32, 1), 
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensities', 12, PointField.FLOAT32, 1)]
    pc = pypcd.PointCloud.from_path(pcdFilePath)
    points = pc.pc_data
    header = std_msgs.msg.Header()
    header.stamp = rospy.rostime.Time.from_sec(time.time())
    header.frame_id = 'velo_link'
    pointCloudMsg = pcl2.create_cloud(header, fields, points)
    return pointCloudMsg

# Converts point cloud data from .bin format to .pcd and writes it 
def convert_bin_to_pcd(binFilePath, outputPcdFileName):
    size_float = 4
    list_pcd = []
    list_pcd_i = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, i = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            list_pcd_i.append(i)
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    np_pcd_i = np.asarray(list_pcd_i)
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np_pcd, dtype, device)
    pcd.point.intensities = o3d.core.Tensor(np_pcd_i.reshape(-1,1), dtype, device)
    o3d.t.io.write_point_cloud(outputPcdFileName, pcd, write_ascii=True)
    return pcd

# Writes pointcloud messages to bag on given topic
def write_pointclouds_to_ros_bag(data, topicName, outputBagName):
    bag = rosbag.Bag(outputBagName, 'w')
    try:
        for msg in data:
            timestamp = rospy.rostime.Time.from_sec(time.time())
            bag.write('/points_raw', msg, timestamp)        
    finally:
        bag.close() 

# Writes image messages to bag on given topic
def write_images_to_ros_bag(imgs, topicName, outputBagName):
    bag = rosbag.Bag(outputBagName, 'w')
    try:
        i = 0
        count = 0
        while(i<len(imgs)):
            img = cv2.imread(imgs[i])
            bridge = CvBridge()
            Stamp = rospy.rostime.Time.from_sec(time.time())
            img_msg = Image()
            img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.seq = i
            img_msg.header.stamp = Stamp
            img_msg.header.frame_id = "camera"
            bag.write(topicName, img_msg, Stamp)
            count += 1
            if(count == 10):
                i += 1
                count = 0  
    finally:
        bag.close()    

# Filters point cloud messages in given rosbag and publishes the filtered messages to given topic
def bounding_box(points, min_x, max_x, min_y, max_y, min_z, max_z):
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    return bb_filter
def filterPointClouds(bagPath, topic, xmin, xmax, ymin, ymax, zmin, zmax, tmin, tmax, publishingTopic):
    bag = rosbag.Bag(bagPath)
    final = []
    pub = rospy.Publisher(publishingTopic, PointCloud2)
    rospy.init_node('pointcloudpublisher', anonymous=True)
    for topic, msg, t in bag.read_messages(topics=[topic]):
        tInSec = t.to_sec()
        if(tInSec > tmin and tInSec < tmax):
            points_with_void = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
            points = np.array(points_with_void.tolist())
            eligibility = bounding_box(points, xmin, xmax, ymin, ymax, zmin, zmax)
            filteredPoints = points[eligibility]
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensities', 12, PointField.FLOAT32, 1)]
            filteredPointsMsg = pcl2.create_cloud(msg.header, fields, filteredPoints)
            final.append(filteredPointsMsg)
            pub.publish(filteredPointsMsg)
    bag.close()
    return np.array(final, dtype='object')
