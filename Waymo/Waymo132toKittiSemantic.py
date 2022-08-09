#from __future__ import absolute_import
from pathlib import Path
import os
import time
from glob import glob
#import matplotlib.pyplot as plt
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
# from waymo_open_dataset import dataset_pb2
# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils


from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
# from waymo_open_dataset.protos import segmentation_metrics_pb2
# from waymo_open_dataset.protos import segmentation_submission_pb2

# Object type definition in: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto enum Type { TYPE_UNKNOWN = 0; TYPE_VEHICLE = 1; TYPE_PEDESTRIAN = 2; TYPE_SIGN = 3; TYPE_CYCLIST = 4; }
objecttype_list = [
    'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
]
INSTANCE_Color = {
    'UNKNOWN': 'black', 'VEHICLE': 'red', 'PEDESTRIAN': 'green', 'SIGN': 'yellow', 'CYCLIST': 'purple'
}  # 'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
lidar_list = [
    '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
    '_SIDE_LEFT'
]

SAVE_IMAGE_LIDAR_TOFILE = True

def save_image(image, fullfolderpath, file_idx, frame_idx):
    img_path = fullfolderpath + \
        f'{str(file_idx).zfill(3)}' + \
        f'{str(frame_idx).zfill(3)}.png'
    if not os.path.exists(fullfolderpath):
        os.makedirs(fullfolderpath)
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_lidar(points, output_path, file_idx, frame_idx):
    # Convert (range, intensity, elongation, x, y, z) to x,y,z,intensity
    # declare new index list
    i = [3, 4, 5, 1]
    # create output
    pointsxyzintensity_output = points[:, i]
    pc_filename = f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
    pcoutput_path = os.path.join(output_path, 'velodyne')
    if not os.path.exists(pcoutput_path):
        os.makedirs(pcoutput_path)
    pointsxyzintensity_output.astype(np.float32).tofile(os.path.join(pcoutput_path, pc_filename))


def extract_frame_images(frame, output_path, file_idx, frame_idx):
    data_dict = {}
    data_dict['POSE'] = np.reshape(
        np.array(frame.pose.transform, np.float32), (4, 4))
    data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

    for img in frame.images:  # go through all images, each frame has 5 images
        cam_name_str = open_dataset.CameraName.Name.Name(
            img.name)  # FRONT, FRONT_LEFT
        image = tf.io.decode_jpeg(
            img.image).numpy()  # tf.image.decode_jpeg(img.image).numpy()

        if SAVE_IMAGE_LIDAR_TOFILE:
            # Save image to file
            foldername = f'image_{str(img.name - 1)}/'  # start 0
            fullfolderpath = os.path.join(output_path, foldername)
            save_image(image, fullfolderpath, file_idx, frame_idx)
        else:
            data_dict[f'{cam_name_str}_IMAGE'] = image
        
        data_dict[f'{cam_name_str}_POSE'] = np.reshape(
            np.array(img.pose.transform, np.float32), (4, 4))

        # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
        for c in frame.context.camera_calibrations:
            # Ignore camera labels that do not correspond to this camera.
            if c.name != img.name:
                continue
            # cam_name_str = open_dataset.CameraName.Name.Name(c.name)
            # print(f'Camera name: {cam_name_str}, width: {c.width}, height: {c.height}')
            data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(
                c.intrinsic, np.float32)
            data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
                np.array(c.extrinsic.transform, np.float32), [4, 4])
            data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
            data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
            # data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DIRECTION'] = np.array(
            #     c.rolling_shutter_direction)

        for camera_label in frame.camera_labels:
            # Ignore camera labels that do not correspond to this camera.
            if camera_label.name != img.name:
                continue

            # Iterate over the individual labels.
            objbboxs = []
            for label in camera_label.labels:
                # print(label.type) #1 for vehicle
                objectclass = objecttype_list[label.type]
                colorlabel = INSTANCE_Color[objectclass]
                # print(label.id) #1fa40b66-1897-4d0b-93e9-a9445372962b
                xmin = label.box.center_x - 0.5 * label.box.length
                ymin = label.box.center_y - 0.5 * label.box.width
                width = label.box.length
                height = label.box.width
                objbbox = [label.type,
                           label.box.center_x - label.box.length / 2,
                           label.box.center_y - label.box.width / 2,
                           label.box.center_x + label.box.length / 2,
                           label.box.center_y + label.box.width / 2
                           ]
                objbboxs.append(objbbox)
            data_dict[f'{cam_name_str}_camera_label'] = np.array(objbboxs)
    return data_dict


def get_3Dbox(frame):
    # The relation between waymo and kitti coordinates is noteworthy:
    #     1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
    #     2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
    #     3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
    #     4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)
    id_to_bbox = dict()
    id_to_name = dict()
    newlabels = []
    for labels in frame.projected_lidar_labels:
        name = labels.name  # camera name
        for label in labels.labels:
            # TODO: need a workaround as bbox may not belong to front cam
            bbox = [
                label.box.center_x - label.box.length / 2,
                label.box.center_y - label.box.width / 2,
                label.box.center_x + label.box.length / 2,
                label.box.center_y + label.box.width / 2
            ]
            id_to_bbox[label.id] = bbox
            id_to_name[label.id] = name - 1

    for obj in frame.laser_labels:
        bounding_box = None
        name = None
        id = obj.id
        for lidar in lidar_list:
            # print("Lidar:", lidar)
            # print("boundingbox:", id_to_bbox.get(id))
            # print("boundingbox id lidar:", id_to_bbox.get(id + lidar))
            #the objid in projected_lidar_labels is idxxx_LidarName
            if id + lidar in id_to_bbox:
                #print("id + lidar:",id + lidar)
                bounding_box = id_to_bbox.get(id + lidar)
                #print("bounding_box:", bounding_box)
                name = str(id_to_name.get(id + lidar))
                # print("name:",name)
                break

        if bounding_box is None or name is None:
            name = '0'
            bounding_box = (0, 0, 0, 0)

        if obj.num_lidar_points_in_box < 1:
            continue

        # objecttype_list[obj.type]
        objtype = obj.type  # in Int
        height = obj.box.height
        width = obj.box.width
        length = obj.box.length
        x = obj.box.center_x
        y = obj.box.center_y
        z = obj.box.center_z
        heading = obj.box.heading
        obj_id = obj.id
        bounding_box[0]  # 0-3 xmin, ymin, xmax, ymax
        newlabel = [objtype, obj_id, bounding_box[0], bounding_box[1],
                    bounding_box[2], bounding_box[3], height, width, length, x, y, z, heading]
        newlabels.append(newlabel)
    return newlabels


def get_rangeimage(range_images, laser_name, return_index):
    rangeimage1 = range_images[laser_name][return_index]
    range_image_tensor = tf.convert_to_tensor(rangeimage1.data)
    range_image_tensor = tf.reshape(range_image_tensor, rangeimage1.shape.dims)
    # Returns the truth value of (x >= y) element-wise.
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(
        lidar_image_mask, range_image_tensor, tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[..., 0]  # only get the range
    range_image_intensity = range_image_tensor[..., 1]
    # Lidar Elongation: Lidar elongation refers to the elongation of the pulse beyond its nominal width. Returns with long pulse elongation, for example, indicate that the laser reflection is potentially smeared or refracted, such that the return pulse is elongated in time.
    range_image_elongation = range_image_tensor[..., 2]
    return range_image_tensor.numpy()


def get_segmentation(segmentation_labels, laser_name, return_index):
    semseg_label_image = segmentation_labels[laser_name][return_index]
    semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
    semseg_label_image_tensor = tf.reshape(
        semseg_label_image_tensor, semseg_label_image.shape.dims)  # [64, 2650, 2]
    # Inner dimensions are [instance_id, semantic_class].
    instance_id_image = semseg_label_image_tensor[..., 0]
    semantic_class_image = semseg_label_image_tensor[..., 1]
    #     3D segmentation labels for 23 classes of 1,150 segments of the Waymo Open Dataset.
    # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/segmentation.proto
    return semseg_label_image_tensor.numpy()


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(
                tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(
                sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(
                tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def extract_frame_lidars(frame, data_dict, output_folder, file_idx, frame_idx):
    # Laser name definition in https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto TOP = 1; FRONT = 2; SIDE_LEFT = 3; SIDE_RIGHT = 4; REAR = 5; The dataset contains data from five lidars - one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear), ref: https://waymo.com/open/data/perception/ The point cloud of each lidar is encoded as a range image. Two range images are provided for each lidar, one for each of the two strongest returns. It has 4 channels:
    # channel 0: range (see spherical coordinate system definition) channel 1: lidar intensity channel 2: lidar elongation channel 3: is_in_nlz (1 = in, -1 = not in) is in any no label zone.
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(
            frame)  # segmentation_labels is new
    # range_images: A dict of {laser_name,
    #   [range_image_first_return, range_image_second_return]}.
    #lasername: 1 [64, 2650, 6]; laser2-5 [200,600,4]

    # camera_projections: A dict of {laser_name,
    #   [camera_projection_from_first_return,
    #   camera_projection_from_second_return]}.
    #lasername: 1 [64, 2650, 6]; laser2-5 [200,600,6]

    # seg_labels: segmentation labels, a dict of {laser_name,
    #   [seg_label_first_return, seg_label_second_return]}

    # range_image_top_pose: range image pixel pose for top lidar. [64, 2650, 6]

    laser_name = open_dataset.LaserName.TOP  # 1
    return_index = 0
    # range_image_np = get_rangeimage(range_images, laser_name, return_index)
    laser_name_str = open_dataset.LaserName.Name.Name(laser_name)  # FRONT
    semseg_label_np = get_segmentation(
        segmentation_labels, laser_name, return_index)
    data_dict[f'{laser_name_str}_semseg_0'] = semseg_label_np

    # Convert range images from polar coordinates to Cartesian coordinates
    # dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
    # will be 3 if keep_polar_features is False (x, y, z) and 6 if
    # keep_polar_features is True (range, intensity, elongation, x, y, z).
    first_return_cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index=0,
        keep_polar_features=True)

    for c in frame.context.laser_calibrations:
        laser_name_str = open_dataset.LaserName.Name.Name(c.name)  # FRONT

        # beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
        # if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
        #     data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
        #         tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
        #         height=range_images[c.name][0].shape.dims[0]).numpy()
        # else:
        #     data_dict[beam_inclination_key] = np.array(
        #         c.beam_inclinations, np.float32)

        data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
            np.array(c.extrinsic.transform, np.float32), [4, 4])

        return_index = 0
        range_image_np = get_rangeimage(range_images, c.name, return_index)
        data_dict[f'{laser_name_str}_RANGE_IMAGE_0'] = range_image_np
        data_dict[f'{laser_name_str}_RANGE_IMAGE_cartesian_0'] = (
            first_return_cartesian_range_images[c.name].numpy())

        # semseg_label_np = get_segmentation(
        # segmentation_labels, c.name, return_index)
        # data_dict[f'{laser_name_str}_semseg_0'] = semseg_label_np
        # data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        #     second_return_cartesian_range_images[c.name].numpy())

    # convert_range_image_to_point_cloud
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True)
    # len(points)#5 lidars, (152136, 6), (3311, 6)
    # points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
    #   (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    # cp_points: {[N, 6]} list of camera projections of length 5
    #   (number of lidars).

    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels)
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1)

    # 3d points in vehicle frame.
    #(range, intensity, elongation, x, y, z)
    # combines 5 lidar data together
    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # point labels.
    point_labels_all = np.concatenate(point_labels, axis=0)
    point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    
    data_dict['point_labels_all'] = point_labels_all
    data_dict['cp_points_all'] = cp_points_all
    if SAVE_IMAGE_LIDAR_TOFILE:
        save_lidar(points_all, output_folder, file_idx, frame_idx)
    else:
        data_dict['points_all'] = points_all
    return data_dict


def extract_onesegment_toKittiSemantic(fileidx, tfrecord_pathnames, step, base_folder):
    segment_path = tfrecord_pathnames[fileidx]
    c_start = time.time()
    print(
        f'extracting {fileidx}, path: {segment_path}, currenttime: {c_start}')

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')

    Final_array = []
    new_frame_idx = 0
    for i, data in enumerate(dataset):
        if i % step != 0:  # Downsample
            continue

        # print('.', end='', flush=True) #progress bar
        frame = open_dataset.Frame()  # dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # check segmentation label available
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            # frames.append(frame)
            print("frame_idx:", i)
            context_name = frame.context.name
            # 14824622621331930560_2395_420_2415_420, same to the tfrecord file
            print('context_name:', context_name)
            framelocation = frame.context.stats.location
            print('frame_location:', framelocation)

            #each tfrecord file (content) will have one folder
            save_folder = base_folder / context_name #os.path.join(base_folder, context_name)

            data_dict = extract_frame_images(
                frame, save_folder, fileidx, new_frame_idx)
            data_dict['context_name'] = context_name
            data_dict['frame_location'] = framelocation

            data_dict = extract_frame_lidars(frame, data_dict, save_folder, fileidx, new_frame_idx)
            new3dlabels = get_3Dbox(frame)
            data_dict['3D_labels']=np.array(new3dlabels)

            frame_timestamp_micros = str(frame.timestamp_micros)
            data_dict['frame_timestamp_micros'] = frame_timestamp_micros
            # print(frame_timestamp_micros)
            Final_array.append(data_dict)
            np.savez_compressed(save_folder/'alldictsnp', data_dict)

            new_frame_idx += 1

    filename = str(prefix)+context_name  # +'.npy'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / filename, Final_array)

    del frame
    del data_dict
    del Final_array


if __name__ == "__main__":
    folders = ["training_0000"]
    root_path = "/data/cmpe249-fa22/Waymo132/"
    out_dir = "/data/cmpe249-fa22/Waymo132KittiSematic2"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files = [path for x in folders for path in glob(
        os.path.join(root_path, x, "*.tfrecord"))]
    print("totoal number of files:", len(data_files))  # 886
    prefix = 'train0_'
    step = 1
    for fileidx in range(len(data_files)):
        extract_onesegment_toKittiSemantic(
            fileidx, data_files, step, out_dir)
    print("finished")
