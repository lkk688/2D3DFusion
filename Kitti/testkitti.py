#from . import pykitti
import pykitti
import numpy as np
import open3d
import time
import argparse
#ref: https://github.com/intel-isl/Open3D-PointNet2-Semantic3D/blob/master/kitti_visualize.py

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti_root", default="/DATA5T/Dataset/Kitti/raw", help="Kitti root file", required=True
    )
    flags = parser.parse_args()

    basedir = flags.kitti_root
    date = "2011_09_26"
    drive = "0001"

    pcd = open3d.PointCloud()
    vis = open3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 0.01

    #data = pykitti.raw(basedir, date, drive)
    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically. 
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))
    # dataset.calib:         Calibration data are accessible as a named tuple
    # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
    # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
    # dataset.camN:          Returns a generator that loads individual images from camera N
    # dataset.get_camN(idx): Returns the image from camera N at idx  
    # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
    # dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
    # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
    # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
    # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
    # dataset.get_velo(idx): Returns the velodyne scan at idx  

    to_reset_view_point = True
    for points_with_intensity in data.velo:
        points = points_with_intensity[:, :3]
        pcd.points = open3d.Vector3dVector(points)

        vis.update_geometry()
        if to_reset_view_point:
            vis.reset_view_point(True)
            to_reset_view_point = False
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)

    vis.destroy_window()







# point_velo = np.array([0,0,0,1])
# point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

# point_imu = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

# for cam0_image in data.cam0:
#     # do something
#     pass

# cam2_image, cam3_image = data.get_rgb(3)
# print(cam2_image.size)
