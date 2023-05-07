#ref: https://github.com/AIR-THU/DAIR-V2X/blob/main/v2x/dataset/dataset_utils/

import json
import yaml
import pickle
import numpy as np
from pypcd import pypcd
#import mmcv #https://mmcv-jm.readthedocs.io/en/latest/image.html
from skimage import io
import os
import errno


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f)
    return data


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)


def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    time = None
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points, time


def read_jpg(jpg_path):
    #image = mmcv.imread(jpg_path)
    assert jpg_path.exists()
    image = io.imread(jpg_path)
    image = image.astype(np.float32)
    image /= 255.0
    return image


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def mdkir_kitti(target_root):
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    os.system("mkdir -p %s/training" % target_root)
    os.system("mkdir -p %s/training/calib" % target_root)
    os.system("mkdir -p %s/training/label_2" % target_root)
    os.system("mkdir -p %s/testing" % target_root)
    os.system("mkdir -p %s/ImageSets" % target_root)
    
def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json


def write_json(path_json, new_dict):
    with open(path_json, "w") as f:
        json.dump(new_dict, f)


def write_txt(path, file):
    with open(path, "w") as f:
        f.write(file)


def get_files_path(path_my_dir, extention=".json"):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(path_my_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def pcd2bin(pcd_file_path, bin_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    points_32.tofile(bin_file_path)

