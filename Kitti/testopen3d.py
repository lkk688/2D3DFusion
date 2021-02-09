import numpy as np
import open3d as open3d
import time
import argparse
import os

#ref: https://github.com/intel-isl/Open3D-PointNet2-Semantic3D/blob/master/kitti_visualize.py

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root", default="/DATA5T/Datasets/Kitti/training/velodyne", help="Kitti root file"
    )
    parser.add_argument(
        "--lidarfile", default="000000.bin", help="Kitti lidar file"
    )
    flags = parser.parse_args()

    basedir = flags.kitti_root
    lidarfile = flags.lidarfile
    fulllidarfilepath = os.path.join(basedir, lidarfile)

    pcd = open3d.geometry.PointCloud()
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 0.01

    to_reset_view_point = True
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(fulllidarfilepath, dtype=np.float32)
    points_with_intensity=scan.reshape((-1, 4))
    points = points_with_intensity[:, :3]
    pcd_points = open3d.utility.Vector3dVector(points)

    open3d.visualization.draw_geometries([pcd_points], zoom=0.8)

    # vis.update_geometry()
    # if to_reset_view_point:
    #     vis.reset_view_point(True)
    #     to_reset_view_point = False
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(0.2)

