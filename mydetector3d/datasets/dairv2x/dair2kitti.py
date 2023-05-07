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

parser = argparse.ArgumentParser("Generate the Kitti Format Data")
parser.add_argument("--source-root", type=str, default="/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/vehicle-side/", help="Raw data root about DAIR-V2X.")
parser.add_argument(
    "--target-root",
    type=str,
    default="/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti",
    help="The data root where the data with kitti format is generated",
)
parser.add_argument(
    "--split-path",
    type=str,
    default="/data/cmpe249-fa22/DAIR-C/split_datas/single-vehicle-split-data.json",
    help="Json file to split the data into training/validation/testing.",
)
parser.add_argument("--label-type", type=str, default="lidar", help="label type from ['lidar', 'camera']")
parser.add_argument("--sensor-view", type=str, default="vehicle", help="Sensor view from ['infrastructure', 'vehicle']")
parser.add_argument(
    "--no-classmerge",
    action="store_true",
    help="Not to merge the four classes [Car, Truck, Van, Bus] into one class [Car]",
)
parser.add_argument("--temp-root", type=str, default="/data/cmpe249-fa22/DAIR-C/tmp_file", help="Temporary intermediate file root.")

def gen_ImageSet_from_split_data(ImageSets_path, split_data_path, sensor_view="vehicle"):
    split_data = read_json(split_data_path)
    test_file = ""
    train_file = ""
    val_file = ""

    if "vehicle_split" in split_data.keys():
        sensor_view = sensor_view + "_split"
        split_data = split_data[sensor_view]
    for i in range(len(split_data["train"])):
        name = split_data["train"][i] #file id
        train_file = train_file + name + "\n" #ids seperated by '\n'

    for i in range(len(split_data["val"])):
        name = split_data["val"][i]
        val_file = val_file + name + "\n"

    # The test part of the dataset has not been released
    # for i in range(len(split_data["test"])):
    #     name = split_data["test"][i]
    #     test_file = test_file + name + "\n"

    trainval_file = train_file + val_file

    mkdir_p(ImageSets_path)
    write_txt(os.path.join(ImageSets_path, "test.txt"), test_file)
    write_txt(os.path.join(ImageSets_path, "trainval.txt"), trainval_file)
    write_txt(os.path.join(ImageSets_path, "train.txt"), train_file)
    write_txt(os.path.join(ImageSets_path, "val.txt"), val_file)

def convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam):
    P2 = np.zeros([3, 4])
    P2[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
    P2 = P2.reshape(12, order="C")

    Tr_velo_to_cam = np.concatenate((r_velo2cam, t_velo2cam), axis=1)
    Tr_velo_to_cam = Tr_velo_to_cam.reshape(12, order="C")

    return P2, Tr_velo_to_cam


def get_cam_D_and_cam_K(path):
    my_json = read_json(path)
    cam_D = my_json["cam_D"]
    cam_K = my_json["cam_K"]
    return cam_D, cam_K


def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = my_json["translation"]
    r_velo2cam = my_json["rotation"]
    return t_velo2cam, r_velo2cam


def gen_calib2kitti(path_camera_intrisinc, path_lidar_to_camera, path_calib):
    path_list_camera_intrisinc = get_files_path(path_camera_intrisinc, ".json")
    path_list_lidar_to_camera = get_files_path(path_lidar_to_camera, ".json")
    path_list_camera_intrisinc.sort()
    path_list_lidar_to_camera.sort()
    print(len(path_list_camera_intrisinc), len(path_list_lidar_to_camera))
    mkdir_p(path_calib)

    for i in range(len(path_list_camera_intrisinc)):
        cam_D, cam_K = get_cam_D_and_cam_K(path_list_camera_intrisinc[i])
        t_velo2cam, r_velo2cam = get_velo2cam(path_list_lidar_to_camera[i])
        json_name = os.path.split(path_list_camera_intrisinc[i])[-1][:-5] + ".txt"
        json_path = os.path.join(path_calib, json_name)

        t_velo2cam = np.array(t_velo2cam).reshape(3, 1)
        r_velo2cam = np.array(r_velo2cam).reshape(3, 3)
        P2, Tr_velo_to_cam = convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam)

        str_P2 = "P2: "
        str_Tr_velo_to_cam = "Tr_velo_to_cam: "
        for ii in range(11):
            str_P2 = str_P2 + str(P2[ii]) + " "
            str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[ii]) + " "
        str_P2 = str_P2 + str(P2[11])
        str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[11])

        str_P0 = str_P2
        str_P1 = str_P2
        str_P3 = str_P2
        str_R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
        str_Tr_imu_to_velo = str_Tr_velo_to_cam

        with open(json_path, "w") as fp:
            gt_line = (
                str_P0
                + "\n"
                + str_P1
                + "\n"
                + str_P2
                + "\n"
                + str_P3
                + "\n"
                + str_R0_rect
                + "\n"
                + str_Tr_velo_to_cam
                + "\n"
                + str_Tr_imu_to_velo
            )
            fp.write(gt_line)

def get_label(label):
    h = float(label["3d_dimensions"]["h"])
    w = float(label["3d_dimensions"]["w"])
    length = float(label["3d_dimensions"]["l"])
    x = float(label["3d_location"]["x"])
    y = float(label["3d_location"]["y"])
    z = float(label["3d_location"]["z"])
    rotation_y = float(label["rotation"])
    return h, w, length, x, y, z, rotation_y


def set_label(label, h, w, length, x, y, z, alpha, rotation_y):
    label["3d_dimensions"]["h"] = h
    label["3d_dimensions"]["w"] = w
    label["3d_dimensions"]["l"] = length
    label["3d_location"]["x"] = x
    label["3d_location"]["y"] = y
    label["3d_location"]["z"] = z
    label["alpha"] = alpha
    label["rotation_y"] = rotation_y


def normalize_angle(angle):
    # make angle in range [-0.5pi, 1.5pi]
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan


def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    ) #(3, 3)
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )#(3, 8)
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T #(3,8)
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam #(3,8)

    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)

    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])

    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi

    alpha_arctan = normalize_angle(alpha)

    return alpha_arctan, yaw


def convert_point(point, matrix):
    return matrix @ point


def get_lidar2cam(calib):
    r_velo2cam = np.array(calib["rotation"])
    t_velo2cam = np.array(calib["translation"])
    r_velo2cam = r_velo2cam.reshape(3, 3)
    t_velo2cam = t_velo2cam.reshape(3, 1)
    return r_velo2cam, t_velo2cam


def gen_lidar2cam(source_root, target_root, label_type="lidar"):
    path_data_info = os.path.join(source_root, "data_info.json")
    data_info = read_json(path_data_info)
    write_path = os.path.join(target_root, "label", label_type) #
    mkdir_p(write_path) #/data/cmpe249-fa22/DAIR-C/tmp_file/label/lidar

    for data in data_info:
        if "calib_virtuallidar_to_camera_path" in data.keys():
            calib_lidar2cam_path = data["calib_virtuallidar_to_camera_path"]
        else:
            calib_lidar2cam_path = data["calib_lidar_to_camera_path"] #calib/lidar_to_camera/000000.json
        calib_lidar2cam = read_json(os.path.join(source_root, calib_lidar2cam_path))
        r_velo2cam, t_velo2cam = get_lidar2cam(calib_lidar2cam)
        Tr_velo_to_cam = np.hstack((r_velo2cam, t_velo2cam)) #(3, 4)

        labels_path = data["label_" + label_type + "_std_path"] #label/lidar/000000.json
        labels = read_json(os.path.join(source_root, labels_path)) #size18 array
        for label in labels:
            h, w, l, x, y, z, yaw_lidar = get_label(label)
            z = z - h / 2
            bottom_center = [x, y, z]
            obj_size = [l, w, h]

            bottom_center_in_cam = r_velo2cam * np.matrix(bottom_center).T + t_velo2cam #(3,1)
            alpha, yaw = get_camera_3d_8points(
                obj_size, yaw_lidar, bottom_center, bottom_center_in_cam, r_velo2cam, t_velo2cam
            )
            cam_x, cam_y, cam_z = convert_point(np.array([x, y, z, 1]).T, Tr_velo_to_cam)

            set_label(label, h, w, l, cam_x, cam_y, cam_z, alpha, yaw)

        labels_path = labels_path.replace("virtuallidar", "lidar") #'label/lidar/000000.json'
        write_path = os.path.join(target_root, labels_path) #/data/cmpe249-fa22/DAIR-C/tmp_file/label/lidar/000000.json

        with open(write_path, "w") as f:
            json.dump(labels, f)


def write_kitti_in_txt(my_json, path_txt):
    wf = open(path_txt, "w")
    for item in my_json:
        i1 = str(item["type"]).title()
        i2 = str(item["truncated_state"])
        i3 = str(item["occluded_state"])
        i4 = str(item["alpha"])
        i5, i6, i7, i8 = (
            str(item["2d_box"]["xmin"]),
            str(item["2d_box"]["ymin"]),
            str(item["2d_box"]["xmax"]),
            str(item["2d_box"]["ymax"]),
        )
        # i9, i10, i11 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["w"]), str(item["3d_dimensions"]["l"])
        i9, i11, i10 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["w"]), str(item["3d_dimensions"]["l"])
        i12, i13, i14 = str(item["3d_location"]["x"]), str(item["3d_location"]["y"]), str(item["3d_location"]["z"])
        
        i15 = str(item["rotation"])
        #i15 = str(-eval(item["rotation"])) #eval() arg 1 must be a string, bytes or code object
        item_list = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        item_string = " ".join(item_list) + "\n"
        wf.write(item_string)
    wf.close()


def json2kitti(json_root, kitti_label_root):
    mkdir_p(kitti_label_root)
    jsons_path = get_files_path(json_root, ".json") #list of /data/cmpe249-fa22/DAIR-C/tmp_file/label/lidar/000000.json
    for json_path in jsons_path:
        my_json = read_json(json_path)
        name = json_path.split("/")[-1][:-5] + ".txt"
        path_txt = os.path.join(kitti_label_root, name)
        write_kitti_in_txt(my_json, path_txt)


def rewrite_txt(path):
    with open(path, "r+") as f:
        data = f.readlines()
        find_str1 = "Truck"
        find_str2 = "Van"
        find_str3 = "Bus"
        replace_str = "Car"
        new_data = ""
        for line in data:
            if find_str1 in line:
                line = line.replace(find_str1, replace_str)
            if find_str2 in line:
                line = line.replace(find_str2, replace_str)
            if find_str3 in line:
                line = line.replace(find_str3, replace_str)
            new_data = new_data + line
    os.remove(path)
    f_new = open(path, "w")
    f_new.write(new_data)
    f_new.close()


def rewrite_label(path_file):
    path_list = get_files_path(path_file, ".txt")
    for path in path_list:
        rewrite_txt(path)


def label_filter(label_dir):
    label_dir = label_dir
    files = os.listdir(label_dir)

    for file in files:
        path = os.path.join(label_dir, file)

        lines_write = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                wlh = float(line.split(" ")[9])
                if wlh > 0:
                    lines_write.append(line)

        with open(path, "w") as f:
            f.writelines(lines_write)



def rawdata_copy(source_root, target_root):
    os.system("cp -r %s/image %s/training/image_2" % (source_root, target_root))
    os.system("cp -r %s/velodyne %s/training" % (source_root, target_root))


def kitti_pcd2bin(target_root):
    pcd_dir = os.path.join(target_root, "training/velodyne")
    fileList = os.listdir(pcd_dir)
    for fileName in fileList:
        if ".pcd" in fileName:
            pcd_file_path = pcd_dir + "/" + fileName
            bin_file_path = pcd_dir + "/" + fileName.replace(".pcd", ".bin")
            pcd2bin(pcd_file_path, bin_file_path)

def mykitti_pcd2bin(sourcelidarfolder, target_root):
    pcd_dir = sourcelidarfolder #os.path.join(target_root, "training/velodyne")
    targetlidar_dir = os.path.join(target_root, "training/velodyne")
    fileList = os.listdir(pcd_dir)
    for fileName in fileList:
        if ".pcd" in fileName:
            pcd_file_path = pcd_dir + "/" + fileName
            bin_file_path = targetlidar_dir + "/" + fileName.replace(".pcd", ".bin")
            pcd2bin(pcd_file_path, bin_file_path)

def step1(args, source_root, target_root):
    print("================ Start to Copy Raw Data ================")
    mdkir_kitti(target_root)
    #rawdata_copy(source_root, target_root) #
    #os.system("cp -r %s/cooperative-vehicle-infrastructure-vehicle-side-image %s/training/image_2" % (source_root, target_root))
    #os.system("cp -r %s/velodyne %s/training" % (source_root, target_root))

    #kitti_pcd2bin(target_root) #xyzi
    sourcelidarfolder = '/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-vehicle-side-velodyne'
    mykitti_pcd2bin(sourcelidarfolder, target_root)

    print("================ Start to Generate Label ================")
    temp_root = args.temp_root #'/data/cmpe249-fa22/DAIR-C/tmp_file
    label_type = args.label_type #lidar
    no_classmerge = args.no_classmerge #False
    os.system("mkdir -p %s" % temp_root)
    os.system("rm -rf %s/*" % temp_root)
    gen_lidar2cam(source_root, temp_root, label_type=label_type)
    
if __name__ == "__main__":
    print("================ Start to Convert ================")
    args = parser.parse_args()
    source_root = args.source_root #/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/vehicle-side/
    target_root = args.target_root #/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti
    temp_root = args.temp_root #'/data/cmpe249-fa22/DAIR-C/tmp_file
    label_type = args.label_type #lidar
    no_classmerge = args.no_classmerge #False

    step1(source_root, target_root)

    json_root = os.path.join(temp_root, "label", label_type) #/data/cmpe249-fa22/DAIR-C/tmp_file/label/lidar
    kitti_label_root = os.path.join(target_root, "training/label_2") #/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/label_2
    json2kitti(json_root, kitti_label_root)
    if not no_classmerge:
        rewrite_label(kitti_label_root)
    label_filter(kitti_label_root)

    os.system("rm -rf %s" % temp_root)

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