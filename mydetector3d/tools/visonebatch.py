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

#from mydetector3d.tools.visual_utils.mayavivisualize_utils import visualize_pts, draw_lidar, draw_gt_boxes3d, draw_scenes #, pltlidar_with3dbox

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchpklfile_path", default='/home/lkk/Developer/data/onebatch_1.pkl', help="pkl file path"
    )#'./data/waymokittisample'
    parser.add_argument(
        "--index", default="10", help="file index"
    )
    parser.add_argument(
        "--dataset", default="waymo", help="dataset name" 
    )#waymokitti
    parser.add_argument(
        "--modelname", default="second", help="model name" 
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

    #draw_scenes

    


