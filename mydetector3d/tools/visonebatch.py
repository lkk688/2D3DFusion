import numpy as np
import os
import cv2
import sys
import argparse
import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle 

from mydetector3d.tools.visual_utils.mayavivisualize_utils import visualize_pts, draw_lidar, draw_gt_boxes3d, draw_scenes #, pltlidar_with3dbox

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

    f = open(args.batchpklfile_path, 'r')   # 'r' for reading; can be omitted
    save_dict = pickle.load(f)         # load file content as mydict
    f.close()

    idx = save_dict['idx']
    modelname = save_dict['modelname'] #='myvoxelnext'
    datasetname = save_dict['datasetname'] #='waymokitti'
    batch_dict = save_dict['batch_dict'] #=batch_dict
    pred_dicts = save_dict['pred_dicts'] #=pred_dicts
    annos = save_dict['annos'] #=annos batch_size array, each dict is the Kitti annotation-like format dict (2D box is converted from 3D pred box)

    # #batch_dict data:
        # 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
        # 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
        # Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
        # Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
        # Voxel_num_points: (89196,)
    batch_gtboxes=batch_dict['gt_boxes']
    batch_points=batch_dict['points']
    batch_voxels=batch_dict['Voxels']
    batch_voxelcoords=batch_dict['Voxel_coords']
    batch_voxelnumpoints=batch_dict['Voxel_num_points']

    idxinbatch = 0
    selectidx=batch_points[:,0] == idxinbatch # idx in the left of 4 point feature (xyzr)
    idx_points = batch_points[selectidx, 1:5] #N,4 points
    idx_gtboxes=batch_gtboxes[idxinbatch, :, :].sequeeze()
    


