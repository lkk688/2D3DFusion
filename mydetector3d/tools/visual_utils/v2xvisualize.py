import os
import json
import argparse
import numpy as np
from pypcd import pypcd
import open3d as o3d

def read_pcd(path_pcd):
    pointpillar = o3d.io.read_point_cloud(path_pcd)
    points = np.asarray(pointpillar.points)
    return points


def draw_pcd(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = o3d.utility.Vector3dVector(point_colors)

    # if gt_boxes is not None:
    #     vis = draw_box(vis, gt_boxes, (0, 0, 1))

    # if ref_boxes is not None:
    #     vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

Dataset_root='/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/'    
parser = argparse.ArgumentParser("Convert The Point Cloud from Infrastructure to Ego-vehicle")
parser.add_argument(
    "--source-root",
    type=str,
    default=Dataset_root,
    help="Raw data root about DAIR-V2X-C.",
)
parser.add_argument(
    "--target-root",
    type=str,
    default=Dataset_root+"vic3d-early-fusion/velodyne/lidar_i2v",
    help="The data root where the data with ego-vehicle coordinate is generated",
)

if __name__ == "__main__":
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root

    Lidar_path=Dataset_root+'vehicle-side/velodyne/015344.pcd'
    points = read_pcd(Lidar_path)
    draw_pcd(points)

    Lidar_path=Dataset_root+'infrastructure-side/velodyne/000009.pcd'
    points = read_pcd(Lidar_path)
    draw_pcd(points)