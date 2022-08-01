import numpy as np
import open3d as o3d
import time
import argparse
import os

#ref: https://github.com/intel-isl/Open3D-PointNet2-Semantic3D/blob/master/kitti_visualize.py

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default='/mnt/DATA5T/Kitti/raw/2011_09_28/2011_09_28_drive_0161_sync/velodyne_points/data/', help="root folder"
    )#r"C:\Users\lkk\Documents\Developer\data"
    parser.add_argument(
        "--lidarfile", default="0000000000.bin", help="Kitti lidar file"
    )#lidarxyzintensityright000150.bin
    flags = parser.parse_args()

    basedir = flags.root_path
    lidarfile = flags.lidarfile
    fulllidarfilepath = os.path.join(basedir, lidarfile)

    pcd = o3d.geometry.PointCloud()
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(fulllidarfilepath, dtype=np.float32) # vector points
    points_with_intensity=scan.reshape((-1, 4))#4 points a group
    points = points_with_intensity[:, :3]#115384, 3
    pcd.points = o3d.utility.Vector3dVector(points)

    # print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud("../../TestData/fragment.ply")
    # print(pcd)
    # print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd], front= [ -0.88003385762528208, -0.28455140189466011, 0.38022481391007151 ],
                                               lookat=[ 7.4001910548025727, 4.4334202214413905, -3.2139019834848428 ],
                                                up= [ 0.39702288370590955, -0.0014773629642567123, 0.91780752187619152 ], 
                                                zoom=0.08)


