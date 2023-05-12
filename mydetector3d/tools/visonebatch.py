import numpy as np
import os
import cv2
import sys
import argparse
import os
# import matplotlib
# matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import pickle 
import torch

from mydetector3d.tools.visual_utils.mayavivisualize_utils import boxes_to_corners_3d, visualize_pts, draw_lidar, draw_gt_boxes3d, mydraw_scenes, draw_scenes #, pltlidar_with3dbox

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(obj, dataset='kitti'):
    """ Takes an object3D
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    """
    #x-y-z: front-left-up (waymo) -> x_right-y_down-z_front(kitti)
    # compute rotational matrix around yaw axis (camera coord y pointing to the bottom, thus Yaw axis is rotate y-axis)
    R = roty(obj.ry)

    # 3d bounding box dimensions: x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
    l = obj.l #x
    w = obj.w #z
    h = obj.h #y

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    #print(corners_3d)
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    # if np.any(corners_3d[2, :] < 0.1): #in Kitti, z axis is to the front, if z<0.1 means objs in back of camera
    #     return np.transpose(corners_3d)
    
    return np.transpose(corners_3d)

def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
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
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchpklfile_path", default='/home/lkk/Developer/data/waymo_models_my3dmodel_epoch128_frame_1.pkl', help="pkl file path"
    )#'./data/waymokittisample'
    parser.add_argument(
        "--index", default="10", help="file index"
    )
    parser.add_argument(
        "--dataset", default="waymo", help="dataset name" 
    )#waymokitti
    parser.add_argument(
        "--modelname", default="my3dmodel", help="model name" 
    )#waymokitti
    parser.add_argument(
        "--camera_count", default=1, help="Number of cameras used"
    )
    args = parser.parse_args()

    f = open(args.batchpklfile_path, 'rb')   # if only use 'r' for reading; it will show error: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
    save_dict = pickle.load(f)         # load file content as mydict
    f.close()

    idx = save_dict['idx']
    modelname = save_dict['modelname'] #='myvoxelnext'
    datasetname = save_dict['datasetname'] #='waymokitti'
    batch_dict = save_dict['batch_dict'] #=batch_dict
    pred_dicts = save_dict['pred_dicts'] ##batch size array of record_dict{'pred_boxes'[N,7],'pred_scores'[N],'pred_labels'[N]}
    annos = save_dict['annos'] #=annos batch_size array, each dict is the Kitti annotation-like format dict (2D box is converted from 3D pred box)

    # #batch_dict data:
        # 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
        # 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
        # Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
        # Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
        # Voxel_num_points: (89196,)
    batch_gtboxes=batch_dict['gt_boxes']
    batch_points=batch_dict['points'] #3033181, 6
    batch_voxels=batch_dict['voxels'] #[1502173, 5, 5]
    batch_voxelcoords=batch_dict['voxel_coords']
    batch_voxelnumpoints=batch_dict['voxel_num_points']

    idxinbatch = 1
    selectidx=batch_points[:,0] == idxinbatch # idx in the left of 4 point feature (xyzr)
    idx_points = batch_points[selectidx, 1:5] #N,4 points [191399, 4]
    idx_gtboxes=torch.squeeze(batch_gtboxes[idxinbatch, :, :], 0) #[104, 8]
    print(idx_gtboxes.shape)

    idx_pred_dicts=pred_dicts[idxinbatch]
    pred_boxes = idx_pred_dicts['pred_boxes'] #[295, 7]
    pred_scores = idx_pred_dicts['pred_scores'] #[295]
    pred_labels = idx_pred_dicts['pred_labels']
    print(pred_boxes.shape)
    print(pred_scores)

    threshold = 0.2
    selectbool = pred_scores > threshold
    pred_boxes = pred_boxes[selectbool,:] #[319, 7]->[58, 7]
    pred_scores = pred_scores[selectbool]
    pred_labels = pred_labels[selectbool]

    import mayavi.mlab as mlab
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4] #[0, -40, -3, 70.4, 40, 1]
    if not isinstance(idx_points, np.ndarray):
        idx_points = idx_points.cpu().numpy()
    draw_lidar(idx_points, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=False, drawregion=True, point_cloud_range=point_cloud_range)
    if idx_gtboxes is not None and not isinstance(idx_gtboxes, np.ndarray):
        idx_gtboxes = idx_gtboxes.cpu().numpy()
    #box3d_pts_3d = compute_box_3d(idx_gtboxes) #3d box coordinate=>get 8 points in camera rect, need 3DObject 
    box3d_pts_3d = boxes_to_corners_3d(idx_gtboxes) #[42,8]->(42, 8, 3)
    #colorlabel=INSTANCE3D_Color[obj.type]
    draw_gt_boxes3d(box3d_pts_3d, fig=fig, color=(1, 1, 1), line_width=1, draw_text=False, label=None) #(n,8,3)

    if pred_boxes is not None and not isinstance(pred_boxes, np.ndarray):
        pred_boxes = pred_boxes.cpu().numpy() #(319,7)
    ref_corners3d = boxes_to_corners_3d(pred_boxes)
    draw_gt_boxes3d(ref_corners3d, fig=fig, color=(0, 1, 0), line_width=1, draw_text=False, label=None) #(n,8,3)
    #fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=None, max_num=300)
    mlab.show()

    #mydraw_scenes(idx_points, idx_gtboxes, pred_boxes)

    print('done')

    #draw_scenes

    


