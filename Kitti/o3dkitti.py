#from . import pykitti
import pykitti
import numpy as np
import open3d as o3d
import time
import argparse
import pandas as pd
#based on testkitti.py

import logging
logging.basicConfig(level=logging.DEBUG)

current_max = 5
# * Get the current frame counter from this variable
counter = 0

# Initialize the display
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1080, height=720)

# * Store the point cloud here for updating
pcd = o3d.geometry.PointCloud()

def update_frame():
    """
    - Updated `pcd` -> point cloud data
    - Removes previous bounding boxes (previous frame)
    - Adds new bounding boxes (current frame)
    """
    global velodyne
    points_with_intensity=next(velodyne)
    points = points_with_intensity[:, :3]
    global pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0, 0, 0])  # Black color


def display_next_frame(event=None):
    global counter
    logging.info(f"Currently: {counter}")
    if counter >= current_max:
        return

    counter += 1
    logging.info(f"Moved to {counter}\n-----")
    update_frame()


def display_previous_frame(event=None):
    global counter
    logging.info(f"Currently: {counter}")
    if counter <= 0:
        return

    counter -= 1
    logging.info(f"Moved to {counter}\n-----")
    update_frame()

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root", default="/DATA5T/Datasets/Kitti/raw", help="Kitti root file"
    )
    flags = parser.parse_args()

    basedir = flags.kitti_root
    date = "2011_09_26"
    drive = "0001"
    data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))
    global velodyne
    velodyne = data.velo

    # These functions increment and decrement the current frame counter
    vis.register_key_callback(262, display_next_frame)  # Right arrow key
    vis.register_key_callback(263, display_previous_frame)  # left arrow key

    # Display the data
    update_frame()

    vis.add_geometry(pcd)
    while True:
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()







# point_velo = np.array([0,0,0,1])
# point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

# point_imu = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

# for cam0_image in data.cam0:
#     # do something
#     pass

# cam2_image, cam3_image = data.get_rgb(3)
# print(cam2_image.size)
