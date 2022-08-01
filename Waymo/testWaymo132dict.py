from glob import glob
import time
import os
from pathlib import Path
import numpy as np
import cv2

def parseDictfile(base_dir, filename):
    Final_array=np.load(base_dir / filename, allow_pickle=True, mmap_mode='r')
    data_array=Final_array['arr_0']
    array_len=len(data_array)
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))

    #for frameid in range(array_len):
    frameid=0
    print("frameid:", frameid)
    convertedframesdict = data_array[frameid] #{'key':key, 'context_name':context_name, 'framedict':framedict}
    context_name=convertedframesdict['context_name']
    print("Context name:", context_name)
    for key, value in convertedframesdict.items():
        print(key)
        print(type(value))
    
    #['points']
    points = convertedframesdict['points']
    points_ri2 = convertedframesdict['points_ri2']
    point_labels = convertedframesdict['point_labels']
    point_labels_ri2 = convertedframesdict['point_labels_ri2']
    cp_points = convertedframesdict['cp_points']# camera projection corresponding to each point.
    cp_points_ri2 = convertedframesdict['cp_points_ri2']

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # point labels.
    point_labels_all = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    # For each camera:
    #     <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
    #     <CAMERA_NAME>_INTRINSIC: 9 float32 array
    #     <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
    #     <CAMERA_NAME>_WIDTH: int64 scalar
    #     <CAMERA_NAME>_HEIGHT: int64 scalar
    #     <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
    #     <CAMERA_NAME>_POSE: 4x4 float32 array
    #     <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
    front_image = convertedframesdict['FRONT_IMAGE'] #FRONT_LEFT_IMAGE, SIDE_LEFT_IMAGE, FRONT_RIGHT_IMAGE, SIDE_RIGHT_IMAGE
    saveimagetofile(front_image, frameid, 0, 'outputs', 'imageoutputs')
    

def saveimagetofile(image, frame_idx, file_idx, basepath, foldername):
    fullfolderpath=os.path.join(basepath,foldername)
    img_path = fullfolderpath + \
        f'{str(file_idx).zfill(3)}' + \
        f'{str(frame_idx).zfill(3)}.png'
    if not os.path.exists(fullfolderpath):
        os.makedirs(fullfolderpath)
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    base_dir="/mnt/DATA10T/Datasets/Waymo132/Outdicts" #"/DATA5T2/Datasets/Waymo132/Outdicts/"#
    base_dir = Path(base_dir)
    filename="train01234_12844373518178303651_2140_000_2160_000.npz"#"training0000__10017090168044687777_6380_000_6400_000.npy.npz"

    parseDictfile(base_dir, filename)
    