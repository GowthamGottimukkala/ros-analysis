import numpy as np
import open3d as o3d
import os
import sys




filelist=[]
directory = r'.'
count=0
for filename in os.listdir(directory):
    if filename.endswith(".bin"):
        filelist.append(os.path.join(directory,filename))
filelist.sort()

for file in filelist:
    bin_pcd = np.fromfile(file, dtype=np.float32)
    print(bin_pcd.shape)
    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # Save to whatever format you like
    o3d.io.write_point_cloud(file[:-4]+'.pcd', o3d_pcd)
