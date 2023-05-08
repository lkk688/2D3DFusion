#ref: https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/visualize/vis_utils.py
#!/usr/bin/env python
# coding: utf-8
import os
import json
import errno
import numpy as np
import cv2
import pickle

import os.path as osp
import mayavi.mlab as mlab
import argparse
import math
from pypcd import pypcd


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json


def get_label(label):
    h = float(label[0]["h"])
    w = float(label[0]["w"])
    length = float(label[0]["l"])
    x = float(label[1]["x"])
    y = float(label[1]["y"])
    z = float(label[1]["z"])
    rotation_y = float(label[-1])
    return h, w, length, x, y, z, rotation_y


def get_lidar2cam(calib):
    if "Tr_velo_to_cam" in calib.keys():
        velo2cam = np.array(calib["Tr_velo_to_cam"]).reshape(3, 4)
        r_velo2cam = velo2cam[:, :3]
        t_velo2cam = velo2cam[:, 3].reshape(3, 1)
    else:
        r_velo2cam = np.array(calib["rotation"])
        t_velo2cam = np.array(calib["translation"])
    return r_velo2cam, t_velo2cam


def get_cam_calib_intrinsic(calib_path):
    my_json = read_json(calib_path)
    cam_K = my_json["cam_K"]
    calib = np.zeros([3, 4])
    calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")

    return calib


def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)

        for start, end in line_indices:
            radius = 5
            color = (0, 0, 250)
            thickness = 1
            cv2.circle(img, (corners[start, 0], corners[start, 1]), radius, color, thickness)
            cv2.circle(img, (corners[end, 0], corners[end, 1]), radius, color, thickness)
            color = (0, 255, 0)
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

    return img.astype(np.uint8)


def get_rgb(img_path):
    return cv2.imread(img_path)


def points_cam2img(points_3d, calib_intrinsic, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    points_3d: N x 8 x 3
    calib_intrinsic: 3 x 4
    return: N x 8 x 2
    """
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)
    # assert len(calib_intrinsic.shape) == 2, 'The dimension of the projection' \
    #                                  f' matrix should be 2 instead of {len(calib_intrinsic.shape)}.'
    # d1, d2 = calib_intrinsic.shape[:2]
    # assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
    #         d1 == 4 and d2 == 4), 'The shape of the projection matrix' \
    #                               f' ({d1}*{d2}) is not supported.'
    # if d1 == 3:
    #     calib_intrinsic_expanded = np.eye(4, dtype=calib_intrinsic.dtype)
    #     calib_intrinsic_expanded[:d1, :d2] = calib_intrinsic
    #     calib_intrinsic = calib_intrinsic_expanded

    # previous implementation use new_zeros, new_one yeilds better results

    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(calib_intrinsic, points_4.T.swapaxes(1, 2).reshape(4, -1))
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        return np.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


def compute_corners_3d(dim, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    # R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2]
    z_corners = [h, h, 0, 0, h, h, 0, 0]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)

    return corners_3d


def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    corners_3d = compute_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)

    return corners_3d


def get_cam_8_points(labels, calib_lidar2cam_path):
    """Plot the boundaries of 3D BBox with label on 2D image.

        Args:
            label: h, w, l, x, y, z, rotaion
            image_path: Path of image to be visualized
            calib_lidar2cam_path: Extrinsic of lidar2camera
            calib_intrinsic_path: Intrinsic of camera
            save_path: Save path for visualized images

    ..code - block:: none


                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / oriign    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y

    """
    calib_lidar2cam = read_json(calib_lidar2cam_path)
    r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
    camera_8_points_list = []
    for label in labels:
        h, w, l, x, y, z, yaw_lidar = get_label(label)
        z = z - h / 2
        bottom_center = [x, y, z]
        obj_size = [l, w, h]
        lidar_8_points = compute_box_3d(obj_size, bottom_center, yaw_lidar)
        # lidar_8_points = np.matrix([[x - l / 2, y + w / 2, z + h],
        #                             [x + l / 2, y + w / 2, z + h],
        #                             [x + l / 2, y + w / 2, z],
        #                             [x - l / 2, y + w / 2, z],
        #                             [x - l / 2, y - w / 2, z + h],
        #                             [x + l / 2, y - w / 2, z + h],
        #                             [x + l / 2, y - w / 2, z],
        #                             [x - l / 2, y - w / 2, z]])
        camera_8_points = r_velo2cam * np.matrix(lidar_8_points).T + t_velo2cam
        camera_8_points_list.append(camera_8_points.T)

    return camera_8_points_list


def vis_label_in_img(camera_8_points_list, img_path, path_camera_intrinsic, save_path):
    # dirs_camera_intrisinc = os.listdir(path_camera_intrinsic)
    # # path_list_camera_intrisinc = get_files_path(path_camera_intrinsic, '.json')
    # # path_list_camera_intrinsic.sort()
    #
    # for frame in dirs_camera_intrinsic:
    index = img_path.split("/")[-1].split(".")[0]
    calib_intrinsic = get_cam_calib_intrinsic(path_camera_intrinsic)
    img = get_rgb(img_path)

    cam8points = np.array(camera_8_points_list)
    num_bbox = cam8points.shape[0]

    uv_origin = points_cam2img(cam8points, calib_intrinsic)
    uv_origin = (uv_origin - 1).round()

    plot_rect3d_on_img(img, num_bbox, uv_origin)
    cv2.imwrite(os.path.join(save_path, index + ".png"), img)
    print(index)

    return True


def draw_boxes3d(
    boxes3d, fig, arrows=None, color=(1, 0, 0), line_width=2, draw_text=True, text_scale=(1, 1, 1), color_list=None
):
    """
    boxes3d: numpy array (n,8,3) for XYZs of the box corners
    fig: mayavi figure handler
    color: RGB value tuple in range (0,1), box line color
    line_width: box line width
    draw_text: boolean, if true, write box indices beside boxes
    text_scale: three number tuple
    color_list: RGB tuple
    """
    num = len(boxes3d)
    for n in range(num):
        if arrows is not None:
            mlab.plot3d(
                arrows[n, :, 0],
                arrows[n, :, 1],
                arrows[n, :, 2],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
        b = boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(b[4, 0], b[4, 1], b[4, 2], "%d" % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    return fig


def read_bin(path):
    pointcloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
    print(pointcloud.shape)
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]
    return x, y, z


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)

    x = np.transpose(pcd.pc_data["x"])
    y = np.transpose(pcd.pc_data["y"])
    z = np.transpose(pcd.pc_data["z"])
    return x, y, z


def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T


def read_label_bboxes(label_path):
    with open(label_path, "r") as load_f:
        labels = json.load(load_f)

    boxes = []
    for label in labels:
        obj_size = [
            float(label["3d_dimensions"]["l"]),
            float(label["3d_dimensions"]["w"]),
            float(label["3d_dimensions"]["h"]),
        ]
        yaw_lidar = float(label["rotation"])
        center_lidar = [
            float(label["3d_location"]["x"]),
            float(label["3d_location"]["y"]),
            float(label["3d_location"]["z"]),
        ]

        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        boxes.append(box)

    return boxes


def plot_box_pcd(x, y, z, boxes):
    vals = "height"
    if vals == "height":
        col = z
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mlab.points3d(
        x,
        y,
        z,
        col,  # Values used for Color
        mode="point",
        colormap="spectral",  # 'bone', 'copper', 'gnuplot'
        color=(1, 1, 0),  # Used a fixed (r,g,b) instead
        figure=fig,
    )
    draw_boxes3d(np.array(boxes), fig, arrows=None)

    mlab.axes(xlabel="x", ylabel="y", zlabel="z")
    mlab.show()


def plot_pred_fusion(args):
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(640, 500))
    data_all = load_pkl(osp.join(args.path, "result", id_to_str(args.id) + ".pkl"))
    print(data_all.keys())

    draw_boxes3d(
        np.array(data_all["boxes_3d"]),
        fig,
        color=(32 / 255, 32 / 255, 32 / 255),
        line_width=1,
    )
    draw_boxes3d(
        np.array(data_all["label"]),
        fig,
        color=(0, 0, 255 / 255),
    )
    mlab.show()


def plot_pred_single(args):
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1280, 1000))
    path = args.path
    file = id_to_str(args.id) + ".pkl"

    data_label = load_pkl(osp.join(path, "result", file))
    label_3d_bboxes = data_label["boxes_3d"]
    if len(label_3d_bboxes.shape) != 3:
        label_3d_bboxes = label_3d_bboxes.squeeze(axis=0)

    data_pred = load_pkl(osp.join(path, "preds", file))
    pred_3d_bboxes = data_pred["boxes_3d"]

    draw_boxes3d(label_3d_bboxes, fig, color=(0, 1, 0))  # vis_label
    draw_boxes3d(pred_3d_bboxes, fig, color=(1, 0, 0))  # vis_pred

    mlab.show()


def plot_label_pcd(args):
    pcd_path = args.pcd_path
    x, y, z = read_pcd(pcd_path)

    label_path = args.label_path
    boxes = read_label_bboxes(label_path)

    plot_box_pcd(x, y, z, boxes)


def add_arguments(parser):
    parser.add_argument("--task", type=str, default="single", choices=["fusion", "single", "pcd_label"])
    parser.add_argument("--path", type=str, default="/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--pcd-path", type=str, default="/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/vehicle-side/velodyne/015344.pcd", help="pcd path to visualize")
    parser.add_argument("--label-path", type=str, default="/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/vehicle-side/label/lidar/015344.json", help="label path to visualize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.task == "fusion":
        plot_pred_fusion(args)

    if args.task == "single":
        plot_pred_single(args)

    if args.task == "pcd_label":
        plot_label_pcd(args)