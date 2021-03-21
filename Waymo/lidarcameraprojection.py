import os

import matplotlib.pyplot as plt
import numpy as np
from kittiutils import *


def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.figure(figsize = (20,15))
    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

#     fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
#                       fgcolor=None, engine=None, size=(1000, 500))
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(3000, 1500))

    draw_lidar(imgfov_pc_velo, fig=fig, bgcolor=(0, 0, 0), pts_scale=10)

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Draw boxes
        draw_gt_boxes3d(boxes3d_pts, fig=fig)
    mlab.show()


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img


if __name__ == '__main__':
    Basepath='/mnt/DATA5T/Share/Dataset/Kitti/object/training'
    filename='000007.png'
    image_file = path = os.path.join(Basepath, 'image_2', filename)
    calibration_file = os.path.join(Basepath, 'calib', filename.replace('png', 'txt'))
    label_file = os.path.join(Basepath, 'label_2', filename.replace('png', 'txt'))
    lidar_file = os.path.join(Basepath, 'velodyne', filename.replace('png', 'bin'))

    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape
    print('Image width:', img_width)
    print('Image height:', img_height)

    # Load calibration
    calib = read_calib_file(calibration_file)
    print('labels:')
    calib

    # Load labels
    labels = load_label(label_file)
    print('labels:')
    labels

    # Load Lidar PC
    pc_velo = load_velo_scan(lidar_file)[:, :3]
    print('Lidar data shape:', pc_velo.shape)

    render_image_with_boxes(rgb, labels, calib)
    render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)