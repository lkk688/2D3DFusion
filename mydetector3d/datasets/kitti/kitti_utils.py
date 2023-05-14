import numpy as np
from ...utils import box_utils
from glob import glob
# def filter_otherobjects(annos, map_name_to_kitti):
#     newannots=[]
#     for anno in annos:
#         anno_size=anno['name'].shape[0]
#         for k in range(anno_size):
#             currentname=anno['name'][k]
#             if currentname in map_name_to_kitti.keys():
#                 newannots.append(anno)

def replaceclass_txt(path, find_strs, replace_str):
    with open(path, "r+") as f:
        data = f.readlines()
        new_data = ""
        for line in data:
            for find_str in find_strs:
                if find_str in line:
                    line = line.replace(find_str, replace_str)
            new_data = new_data + line
    os.remove(path)
    f_new = open(path, "w")
    f_new.write(new_data)
    f_new.close()

def rewrite_label2(root_path, folders):
    path_list = [path for x in folders for path in glob(os.path.join(root_path, x, "*.txt"))]
    #path_list = get_files_path(path_file, ".txt")
    #["Truck","Van","Bus","Car"] has been converted to Car in dair2kitti conversion
    find_strs = ["Truck","Van","Bus","Car"]
    replace_str = "Car"
    for path in path_list:
        replaceclass_txt(path, find_strs, replace_str)
        #rewrite_txt(path)

def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            currentname=anno['name'][k]
            if currentname in map_name_to_kitti.keys():
                anno['name'][k] = map_name_to_kitti[currentname]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_trainvaltestsplitfile(dataset_dir, output_dir, imagefoldername='image_2'): #waymo is image_0
    trainingdir = os.path.join(dataset_dir, imagefoldername)
    ImageSetdir = os.path.join(output_dir, 'ImageSets')
    if not os.path.exists(ImageSetdir):
        os.makedirs(ImageSetdir)

    images = os.listdir(trainingdir)
    # totalimages=len([img for img in images])
    # print("Total images:", totalimages)
    dataset = []
    for img in images:
        dataset.append(img[:-4])#remove .png
    print("Total images:", len(dataset))
    df = pd.DataFrame(dataset, columns=['index'])
    X_train, X_val = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)
    print("Train size:", X_train.shape)
    print("Val size:", X_val.shape)
    write_to_file(os.path.join(ImageSetdir, 'trainval.txt'), df['index'])
    write_to_file(os.path.join(ImageSetdir, 'train.txt'), X_train['index'])
    write_to_file(os.path.join(ImageSetdir, 'val.txt'), X_val['index'])

def write_to_file(path, data): 
    file = open(path, 'w') 
    for row in data: 
        #print(idx)
        #file.write(str(idx).zfill(6))
        file.write(row)
        file.write('\n')

    file.close()
    print('Done in ' + path)

# def get_annotationfromlabel(obj_list, calib):
#     name, truncated, occluded, alpha, bbox = [], [], [], [], []
#     dimensions, location, rotation_y, score = [], [], [], []
#     difficulty = []
#     for obj in obj_list:
#         name.append(obj.cls_type)
#         truncated.append(obj.truncation)
#         occluded.append(obj.occlusion)
#         alpha.append(obj.alpha)
#         bbox.append(obj.box2d.reshape(1, 4))
#         dimensions.append([obj.l, obj.h, obj.w])
#         location.append(obj.loc.reshape(1, 3))
#         rotation_y.append(obj.alpha)
#         score.append(obj.score)
#         difficulty.append(obj.level)

#     annotations = {}
#     annotations['name'] = np.array(name)
#     annotations['truncated'] = np.array(truncated)
#     annotations['difficulty'] = np.array(difficulty)
#     annotations['dimensions'] = np.array(dimensions)
#     annotations['location'] = np.array(location)
#     annotations['heading_angles'] = np.array(heading_angles)


#     annotations['obj_ids'] = np.array(obj_ids)
#     annotations['tracking_difficulty'] = np.array(tracking_difficulty)
#     annotations['num_points_in_gt'] = np.array(num_points_in_gt)
#     annotations['speed_global'] = np.array(speeds)
#     annotations['accel_global'] = np.array(accelerations)

#     annotations = {}
#     annotations['name'] = np.array([obj.cls_type for obj in obj_list])
#     annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
#     annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
#     annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
#     annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
#     annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
#     annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
#     annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
#     annotations['score'] = np.array([obj.score for obj in obj_list])
#     annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

#     num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
#     num_gt = len(annotations['name'])
#     index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
#     annotations['index'] = np.array(index, dtype=np.int32)

#     loc = annotations['location'][:num_objects]
#     dims = annotations['dimensions'][:num_objects]
#     rots = annotations['rotation_y'][:num_objects]
#     loc_lidar = calib.rect_to_lidar(loc)
#     l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
#     loc_lidar[:, 2] += h[:, 0] / 2
#     gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
#     annotations['gt_boxes_lidar'] = gt_boxes_lidar