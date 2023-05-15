#ref: https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/dataset_converter/point_cloud_i2v.py
#can convert the point cloud from infrastructure LiDAR coordinate system to ego-vehicle LiDAR coordinate system.

import os
import json
import argparse
import numpy as np
from pypcd import pypcd

from tqdm import tqdm
import errno

from concurrent import futures as futures

# import open3d as o3d

# def read_pcd(path_pcd):
#     pointpillar = o3d.io.read_point_cloud(path_pcd)
#     points = np.asarray(pointpillar.points)
#     return points


# def show_pcd(path_pcd):
#     pcd = read_pcd(path_pcd)
#     o3d.visualization.draw_geometries([pcd])

def read_pcd(path_pcd):
    pc = pypcd.PointCloud.from_path(path_pcd)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    return points_32




def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_virtuallidar2world(path_virtuallidar2world):
    virtuallidar2world = read_json(path_virtuallidar2world)
    rotation = virtuallidar2world["rotation"]
    translation = virtuallidar2world["translation"]
    delta_x = virtuallidar2world["relative_error"]["delta_x"]
    delta_y = virtuallidar2world["relative_error"]["delta_y"]
    return rotation, translation, delta_x, delta_y


def get_novatel2world(path_novatel2world):
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world["rotation"]
    translation = novatel2world["translation"]
    return rotation, translation


def get_lidar2novatel(path_lidar2novatel):
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel["transform"]["rotation"]
    translation = lidar2novatel["transform"]["translation"]
    return rotation, translation


def get_data(data_info, path_pcd):
    for data in data_info:
        name1 = os.path.split(path_pcd)[-1]
        name2 = os.path.split(data["pointcloud_path"])[-1]
        if name1 == name2:
            return data


def trans(input_point, translation, rotation):
    input_point = np.array(input_point).reshape(3, -1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, -1) + np.array(translation).reshape(3, 1)
    return output_point


def rev_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def trans_point_i2v(input_point, path_virtuallidar2world, path_novatel2world, path_lidar2novatel):
    # print('0:', input_point)

    # virtuallidar to world
    rotation, translation, delta_x, delta_y = get_virtuallidar2world(path_virtuallidar2world)
    point = trans(input_point, translation, rotation) + np.array([delta_x, delta_y, 0]).reshape(3, 1) #(3, 54309)
    """
    print('rotation, translation, delta_x, delta_y', rotation, translation, delta_x, delta_y)
    print('1:', point)
    """

    # world to novatel
    rotation, translation = get_novatel2world(path_novatel2world)
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation) #(3, 54309)
    """
    print('rotation, translation:', rotation, translation)
    print('new_translation, new_rotation:', new_translation, new_rotation)
    print('2:', point)
    """

    # novatel to lidar
    rotation, translation = get_lidar2novatel(path_lidar2novatel) #vehicle-side/calib/lidar_to_novatel
    new_rotation = rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point = trans(point, new_translation, new_rotation)
    """
    print('rotation, translation:', rotation, translation)
    print('new_translation, new_rotation:', new_translation, new_rotation)
    print('3:', point)
    """
    point = point.T

    return point #(54309, 3)


def write_pcd(path_pcd, new_points, path_save):
    pc = pypcd.PointCloud.from_path(path_pcd)
    pc.pc_data["x"] = new_points[:, 0]
    pc.pc_data["y"] = new_points[:, 1]
    pc.pc_data["z"] = new_points[:, 2]
    pc.save_pcd(path_save, compression="binary_compressed")

def trans_pcd_i2v(path_pcd, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save):
    points = read_pcd(path_pcd) #infrastructure side lidar pcd (n,4)
    # (n, 3)
    xyzpoints = points[:,0:3]
    intensitypoints = points[:,3]
    new_points = trans_point_i2v(xyzpoints.T, path_virtuallidar2world, path_novatel2world, path_lidar2novatel)
    intensityarray =intensitypoints[:, None] #(n,) and (n,1) are not the same shape. Try casting the vector to an array by using the [:, None] notation:
    new_points4 = np.hstack((new_points, intensityarray))#, axis=1)
    #new_points4.tofile(path_save.replace(".pcd", ".bin"))
    write_pcd(path_pcd, new_points, path_save)

def trans_pcd_i2vnew(path_pcd, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save):
    points = read_pcd(path_pcd) #infrastructure side lidar pcd (n,4)
    # (n, 3)
    xyzpoints = points[:,0:3]
    intensitypoints = points[:,3]
    new_points = trans_point_i2v(xyzpoints.T, path_virtuallidar2world, path_novatel2world, path_lidar2novatel)
    intensityarray =intensitypoints[:, None] #(n,) and (n,1) are not the same shape. Try casting the vector to an array by using the [:, None] notation:
    new_points4 = np.hstack((new_points, intensityarray))#, axis=1)
    new_points4.tofile(path_save.replace(".pcd", ".bin"))
    #write_pcd(path_pcd, new_points, path_save)

    
def map_func(data, path_c, path_dest, i_data_info, v_data_info):
    path_pcd_i = os.path.join(path_c, data["infrastructure_pointcloud_path"])
    path_pcd_v = os.path.join(path_c, data["vehicle_pointcloud_path"])
    i_data = get_data(i_data_info, path_pcd_i) #find the datainfo["pointcloud_path"] matched to the frame path pcd name
    v_data = get_data(v_data_info, path_pcd_v)#only use path_pcd_v file name not path
    path_virtuallidar2world = os.path.join(
        path_c, "infrastructure-side", i_data["calib_virtuallidar_to_world_path"]
    )
    path_novatel2world = os.path.join(path_c, "vehicle-side", v_data["calib_novatel_to_world_path"])
    path_lidar2novatel = os.path.join(path_c, "vehicle-side", v_data["calib_lidar_to_novatel_path"])
    name = os.path.split(path_pcd_i)[-1]
    path_save = os.path.join(path_dest, name) #save to /vic3d-early-fusion/velodyne/lidar_i2v/000049.pcd
    trans_pcd_i2v(path_pcd_i, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save)
    
    
def get_i2v(path_c, path_dest, num_worker):
    mkdir_p(path_dest) #create vic3d-early-fusion folder
    path_c_data_info = os.path.join(path_c, "cooperative/data_info.json")
    path_i_data_info = os.path.join(path_c, "infrastructure-side/data_info.json")
    path_v_data_info = os.path.join(path_c, "vehicle-side/data_info.json")
    c_data_info = read_json(path_c_data_info) #46
    i_data_info = read_json(path_i_data_info) #84
    v_data_info = read_json(path_v_data_info) #91
    
    total = len(c_data_info) #46
    with tqdm(total=total) as pbar:
        with futures.ProcessPoolExecutor(num_worker) as executor:
            res = [executor.submit(map_func, data, path_c, path_dest, i_data_info, v_data_info) for data in c_data_info]
            for _ in futures.as_completed(res):
                pbar.update(1)

def get_i2vnew(path_c, path_dest, infrastructurelidar, num_worker):
    mkdir_p(path_dest) #create early-fusion folder
    path_c_data_info = os.path.join(path_c, "cooperative/data_info.json")
    path_i_data_info = os.path.join(path_c, "infrastructure-side/data_info.json")
    path_v_data_info = os.path.join(path_c, "vehicle-side/data_info.json")
    c_data_info = read_json(path_c_data_info) #46
    i_data_info = read_json(path_i_data_info) #84
    v_data_info = read_json(path_v_data_info) #91
    
    total = len(c_data_info) #46
    # for data in c_data_info:
    #     map_funcnew(data, path_c, path_dest, i_data_info, v_data_info, infrastructurelidar)
    with tqdm(total=total) as pbar:
        with futures.ProcessPoolExecutor(num_worker) as executor:
            res = [executor.submit(map_funcnew, data, path_c, path_dest, i_data_info, v_data_info, infrastructurelidar) for data in c_data_info]
            for _ in futures.as_completed(res):
                pbar.update(1)


def map_funcnew(data, path_c, path_dest, i_data_info, v_data_info, infrastructurelidar):
    infrastructure_pointcloud_path=data["infrastructure_pointcloud_path"]
    path_pcd_i = os.path.join(infrastructurelidar, os.path.split(infrastructure_pointcloud_path)[-1])
    path_pcd_v = os.path.join(path_c, data["vehicle_pointcloud_path"])#actual data not used
    i_data = get_data(i_data_info, path_pcd_i) #find the datainfo["pointcloud_path"] matched to the frame path pcd name
    v_data = get_data(v_data_info, path_pcd_v)
    path_virtuallidar2world = os.path.join(
        path_c, "infrastructure-side", i_data["calib_virtuallidar_to_world_path"]
    )
    path_novatel2world = os.path.join(path_c, "vehicle-side", v_data["calib_novatel_to_world_path"])
    path_lidar2novatel = os.path.join(path_c, "vehicle-side", v_data["calib_lidar_to_novatel_path"])
    name = os.path.split(path_pcd_i)[-1]
    path_save = os.path.join(path_dest, name) #save to /vic3d-early-fusion/velodyne/lidar_i2v/000049.pcd
    trans_pcd_i2vnew(path_pcd_i, path_virtuallidar2world, path_novatel2world, path_lidar2novatel, path_save)
    

#Dataset_root='/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/'    
# Dataset_root='/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/'
Dataset_root='/data/cmpe249-fa22/DAIR-C/'
parser = argparse.ArgumentParser("Convert The Point Cloud from Infrastructure to Ego-vehicle")
parser.add_argument(
    "--source-root",
    type=str,
    default=Dataset_root+'cooperative-vehicle-infrastructure/',
    help="Raw data root about DAIR-V2X-C.",
)
parser.add_argument(
    "--target-root",
    type=str,
    default=Dataset_root+"early-fusion/velodyne/lidar_i2v",
    help="The data root where the data with ego-vehicle coordinate is generated",
)
parser.add_argument(
    "--infrastructurelidar",
    type=str,
    default=Dataset_root+"cooperative-vehicle-infrastructure-infrastructure-side-velodyne",
    help="lidar data path, new added",
)
parser.add_argument(
    "--num-worker",
    type=int,
    default=4,
    help="Number of workers for multi-processing",
)

#Lidar path: /data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-infrastructure-side-velodyne
if __name__ == "__main__":
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root

    num_worker = args.num_worker

    #get_i2v(source_root, target_root, num_worker)
    get_i2vnew(source_root, target_root, args.infrastructurelidar, num_worker)
