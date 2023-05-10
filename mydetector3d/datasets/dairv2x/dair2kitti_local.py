#ref: https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/dataset_converter/dair2kitti.py
import argparse
import os
import numpy as np
import json
import errno
import math
#from gen_kitti.label_lidarcoord_to_cameracoord import gen_lidar2cam
#from gen_kitti.label_json2kitti import json2kitti, rewrite_label, label_filter
#from gen_kitti.gen_calib2kitti import gen_calib2kitti
#from gen_kitti.gen_ImageSets_from_split_data import gen_ImageSet_from_split_data
#from gen_kitti.utils import pcd2bin

from mydetector3d.utils.dataset_utils import load_json, get_files_path, mkdir_p, read_json, mdkir_kitti, \
    write_txt, pcd2bin
from mydetector3d.datasets.dairv2x.dair2kitti import step1, step2_kittilabel, gen_calib2kitti, gen_ImageSet_from_split_data

parser = argparse.ArgumentParser("Generate the Kitti Format Data")
parser.add_argument("--source-root", type=str, default="/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure/infrastructure-side/", help="Raw data root about DAIR-V2X.")
parser.add_argument(
    "--target-root",
    type=str,
    default="/mnt/f/Dataset/DAIR-C/infrastructure-side-point-cloud-kitti",
    help="The data root where the data with kitti format is generated",
)
parser.add_argument(
    "--sourcelidarfolder",
    type=str,
    default="/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-infrastructure-side-velodyne",
    help="The Lidar pcd file location",
)
parser.add_argument(
    "--split-path",
    type=str,
    default="/mnt/f/Dataset/DAIR-C/split_datas/single-infrastructure-split-data.json",
    help="Json file to split the data into training/validation/testing.",
)
parser.add_argument("--label-type", type=str, default="lidar", help="label type from ['lidar', 'camera']")
parser.add_argument("--sensor-view", type=str, default="infrastructure", help="Sensor view from ['infrastructure', 'vehicle']")
parser.add_argument(
    "--no-classmerge",
    action="store_true",
    help="Not to merge the four classes [Car, Truck, Van, Bus] into one class [Car]",
)
parser.add_argument("--temp-root", type=str, default="/mnt/f/Dataset/DAIR-C/tmp_file", help="Temporary intermediate file root.")


if __name__ == "__main__":
    print("================ Start to Convert ================")
    args = parser.parse_args()
    source_root = args.source_root #/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/vehicle-side/
    target_root = args.target_root #/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti
    temp_root = args.temp_root #'/data/cmpe249-fa22/DAIR-C/tmp_file
    label_type = args.label_type #lidar
    no_classmerge = args.no_classmerge #False

    #Get Lidar bin file and label.json
    #step1(args, source_root, target_root, args.sourcelidarfolder)

    #Convert json label to kitti label txt file
    step2_kittilabel(temp_root,label_type,target_root,no_classmerge)

    print("================ Start to Generate Calibration Files ================")
    sensor_view = args.sensor_view
    path_camera_intrinsic = os.path.join(source_root, "calib/camera_intrinsic")
    if sensor_view == "vehicle" or sensor_view == "cooperative":
        path_lidar_to_camera = os.path.join(source_root, "calib/lidar_to_camera") #/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/vehicle-side/calib/lidar_to_camera
    else:
        path_lidar_to_camera = os.path.join(source_root, "calib/virtuallidar_to_camera")
    path_calib = os.path.join(target_root, "training/calib") #/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/calib
    gen_calib2kitti(path_camera_intrinsic, path_lidar_to_camera, path_calib)

    print("================ Start to Generate ImageSet Files ================")
    split_json_path = args.split_path #/data/cmpe249-fa22/DAIR-C/split_datas/single-vehicle-split-data.json
    ImageSets_path = os.path.join(target_root, "ImageSets") #/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/ImageSets
    gen_ImageSet_from_split_data(ImageSets_path, split_json_path, sensor_view)

    #Last step: copy image data to Kitti/image_2 folder