import numpy as np
import open3d as o3d
import os
import sys




filelist=[]
directory = r'.'
count=0
for filename in os.listdir(directory):
    if filename.endswith(".bin") and "3_clusters" in filename:
        filelist.append(os.path.join(directory,filename))
filelist.sort()



for i, file in enumerate(filelist):
    bin_pcd = np.fromfile(file, dtype=np.float32)
    if (i==0):
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
    else:
        points = np.vstack((points, bin_pcd.reshape((-1, 4))[:, 0:3]))


o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))   
o3d.io.write_point_cloud('3_clusters.pcd', o3d_pcd)
