# ref: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/frame_utils.py

#from __future__ import absolute_import
from pathlib import Path
import os
import time
from glob import glob
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf

# from waymo_open_dataset import dataset_pb2
# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils


from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset



def convert_frame_to_dict_lidar(frame):
    """Convert the frame proto into a dict of numpy arrays.
    The keys, shapes, and data types are:
      POSE: 4x4 float32 array
      For each lidar:
        <LIDAR_NAME>_BEAM_INCLINATION: H float32 array
        <LIDAR_NAME>_LIDAR_EXTRINSIC: 4x4 float32 array
        <LIDAR_NAME>_RANGE_IMAGE_FIRST_RETURN: HxWx6 float32 array
        <LIDAR_NAME>_RANGE_IMAGE_SECOND_RETURN: HxWx6 float32 array
        <LIDAR_NAME>_CAM_PROJ_FIRST_RETURN: HxWx6 int64 array
        <LIDAR_NAME>_CAM_PROJ_SECOND_RETURN: HxWx6 float32 array
        (top lidar only) TOP_RANGE_IMAGE_POSE: HxWx6 float32 array

    Args:
      frame: open dataset frame
    Returns:
      Dict from string field name to numpy ndarray.
    """
    # Laser name definition in https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto TOP = 1; FRONT = 2; SIDE_LEFT = 3; SIDE_RIGHT = 4; REAR = 5; The dataset contains data from five lidars - one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear), ref: https://waymo.com/open/data/perception/ The point cloud of each lidar is encoded as a range image. Two range images are provided for each lidar, one for each of the two strongest returns. It has 4 channels:
    # channel 0: range (see spherical coordinate system definition) channel 1: lidar intensity channel 2: lidar elongation channel 3: is_in_nlz (1 = in, -1 = not in)
    range_images, camera_projection_protos, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame))

    # Convert range images from polar coordinates to Cartesian coordinates
    # dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
    # will be 3 if keep_polar_features is False (x, y, z) and 6 if
    # keep_polar_features is True (range, intensity, elongation, x, y, z).
    first_return_cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index=0,
        keep_polar_features=True)

    # second_return_cartesian_range_images = convert_range_image_to_cartesian(
    #     frame, range_images, range_image_top_pose, ri_index=1,
    #     keep_polar_features=True)

    data_dict = {}

    # Save the beam inclinations, extrinsic matrices, first/second return range
    # images, and first/second return camera projections for each lidar.
    for c in frame.context.laser_calibrations:
        laser_name_str = open_dataset.LaserName.Name.Name(c.name)

        # beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
        # if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
        #     data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
        #         tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
        #         height=range_images[c.name][0].shape.dims[0]).numpy()
        # else:
        #     data_dict[beam_inclination_key] = np.array(
        #         c.beam_inclinations, np.float32)

        # data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
        #     np.array(c.extrinsic.transform, np.float32), [4, 4])

        data_dict[f'{laser_name_str}_RANGE_IMAGE_FIRST_RETURN'] = (
            first_return_cartesian_range_images[c.name].numpy())
        # data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        #     second_return_cartesian_range_images[c.name].numpy())

        # first_return_cp = camera_projection_protos[c.name][0]
        # data_dict[f'{laser_name_str}_CAM_PROJ_FIRST_RETURN'] = np.reshape(
        #     np.array(first_return_cp.data), first_return_cp.shape.dims)

        # second_return_cp = camera_projection_protos[c.name][1]
        # data_dict[f'{laser_name_str}_CAM_PROJ_SECOND_RETURN'] = np.reshape(
        #     np.array(second_return_cp.data), second_return_cp.shape.dims)

    # # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
    # for im in frame.images:
    #     cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
    #     data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(
    #         im.image).numpy()

    # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
    # for c in frame.context.camera_calibrations:
    #     cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    #     data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(
    #         c.intrinsic, np.float32)
    #     data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
    #         np.array(c.extrinsic.transform, np.float32), [4, 4])
    #     data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
    #     data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)

    # # Save the range image pixel pose for the top lidar.
    # data_dict['TOP_RANGE_IMAGE_POSE'] = np.reshape(
    #     np.array(range_image_top_pose.data, np.float32),
    #     range_image_top_pose.shape.dims)

    data_dict['POSE'] = np.reshape(
        np.array(frame.pose.transform, np.float32), (4, 4))

    return data_dict


# ref from https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/frame_utils.py
def convert_frame_to_dict_cameras(frame):
    """Convert the frame proto into a dict of numpy arrays.
    The keys, shapes, and data types are:
      POSE: 4x4 float32 array
      TIMESTAMP: int64 scalar
      For each camera:
        <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
        <CAMERA_NAME>_INTRINSIC: 9 float32 array
        <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
        <CAMERA_NAME>_WIDTH: int64 scalar
        <CAMERA_NAME>_HEIGHT: int64 scalar
        <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
        <CAMERA_NAME>_POSE: 4x4 float32 array
        <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
        <CAMERA_NAME>_ROLLING_SHUTTER_DURATION: float32 scalar
        <CAMERA_NAME>_ROLLING_SHUTTER_DIRECTION: int64 scalar
        <CAMERA_NAME>_CAMERA_TRIGGER_TIME: float32 scalar
        <CAMERA_NAME>_CAMERA_READOUT_DONE_TIME: float32 scalar
    NOTE: This function only works in eager mode for now.
    See the LaserName.Name and CameraName.Name enums in dataset.proto for the
    valid lidar and camera name strings that will be present in the returned
    dictionaries.
    Args:
      frame: open dataset frame
    Returns:
      Dict from string field name to numpy ndarray.
    """
    data_dict = {}

    # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
    for im in frame.images:
        cam_name_str = open_dataset.CameraName.Name.Name(im.name)
        data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(
            im.image).numpy()
        # data_dict[f'{cam_name_str}_SDC_VELOCITY'] = np.array([
        #     im.velocity.v_x, im.velocity.v_y, im.velocity.v_z, im.velocity.w_x,
        #     im.velocity.w_y, im.velocity.w_z
        # ], np.float32)
        # data_dict[f'{cam_name_str}_POSE'] = np.reshape(
        #     np.array(im.pose.transform, np.float32), (4, 4))
        # data_dict[f'{cam_name_str}_POSE_TIMESTAMP'] = np.array(
        #     im.pose_timestamp, np.float32)
        # data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(
        #     im.shutter)
        # data_dict[f'{cam_name_str}_CAMERA_TRIGGER_TIME'] = np.array(
        #     im.camera_trigger_time)
        # data_dict[f'{cam_name_str}_CAMERA_READOUT_DONE_TIME'] = np.array(
        #     im.camera_readout_done_time)

    # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
    for c in frame.context.camera_calibrations:
        cam_name_str = open_dataset.CameraName.Name.Name(c.name)
        print(f'Camera name: {cam_name_str}, width: {c.width}, height: {c.height}')
        data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(
            c.intrinsic, np.float32)
        data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
            np.array(c.extrinsic.transform, np.float32), [4, 4])
        data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
        data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
        # data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DIRECTION'] = np.array(
        #     c.rolling_shutter_direction)

    data_dict['POSE'] = np.reshape(
        np.array(frame.pose.transform, np.float32), (4, 4))
    data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

    return data_dict

#from waymo_open_dataset import dataset_pb2 as open_dataset


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
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:#TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)#(64, 2650, 4)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:#TOP, only TOP lidar
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)#(64, 2650, 2)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:#other Lidar
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())#other Lidar does not have segmentation label, put zero
    return point_labels

def processoneframe_todict(frame):
    # get one frame
    # A unique name that identifies the frame sequence
    data_dict = convert_frame_to_dict_cameras(frame)
    

    # Laser name definition in https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto TOP = 1; FRONT = 2; SIDE_LEFT = 3; SIDE_RIGHT = 4; REAR = 5; The dataset contains data from five lidars - one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear), ref: https://waymo.com/open/data/perception/ The point cloud of each lidar is encoded as a range image. Two range images are provided for each lidar, one for each of the two strongest returns. It has 4 channels:
    # channel 0: range (see spherical coordinate system definition) channel 1: lidar intensity channel 2: lidar elongation channel 3: is_in_nlz (1 = in, -1 = not in)
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert range images from polar coordinates to Cartesian coordinates
    # dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
    # will be 3 if keep_polar_features is False (x, y, z) and 6 if
    # keep_polar_features is True (range, intensity, elongation, x, y, z).
    first_return_cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index=0,
        keep_polar_features=True)

    # second_return_cartesian_range_images = convert_range_image_to_cartesian(
    #     frame, range_images, range_image_top_pose, ri_index=1,
    #     keep_polar_features=True)

    # Save the beam inclinations, extrinsic matrices, first/second return range
    # images, and first/second return camera projections for each lidar.
    for c in frame.context.laser_calibrations:
        laser_name_str = open_dataset.LaserName.Name.Name(c.name)#FRONT

        # beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
        # if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
        #     data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
        #         tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
        #         height=range_images[c.name][0].shape.dims[0]).numpy()
        # else:
        #     data_dict[beam_inclination_key] = np.array(
        #         c.beam_inclinations, np.float32)

        # data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
        #     np.array(c.extrinsic.transform, np.float32), [4, 4])

        data_dict[f'{laser_name_str}_RANGE_IMAGE_FIRST_RETURN'] = (
            first_return_cartesian_range_images[c.name].numpy())
        # data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        #     second_return_cartesian_range_images[c.name].numpy())
    
    #Process segmentation_labels
    #print(segmentation_labels[open_dataset.LaserName.TOP][0].shape.dims)#[64, 2650, 2], open_dataset.LaserName.TOP=1
    # laser_name=open_dataset.LaserName.TOP
    # return_index=0
    # semseg_label_image=segmentation_labels[laser_name][return_index]
    # semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
    # semseg_label_image_tensor = tf.reshape(semseg_label_image_tensor, semseg_label_image.shape.dims)#TensorShape([64, 2650, 2])
    # #Inner dimensions are [instance_id, semantic_class].
    # instance_id_image = semseg_label_image_tensor[...,0] 
    # semantic_class_image = semseg_label_image_tensor[...,1]#TensorShape([64, 2650])

    #Process range_images, convert to point cloud
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

    #point_labels: {[N, 2]} list of 3d lidar points's segmentation labels
    point_labels = convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels)#len=5 array, only the first item is the TOP lidar label
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(frame, range_images, segmentation_labels, ri_index=1)

    #len=5 array for 5 lidars, not combined
    data_dict['points'] = points #3d points in vehicle frame.
    data_dict['points_ri2'] = points_ri2
    data_dict['point_labels'] = point_labels # point labels.
    data_dict['point_labels_ri2'] = point_labels_ri2
    data_dict['cp_points'] = cp_points # camera projection corresponding to each point.
    data_dict['cp_points_ri2'] = cp_points_ri2

    # # 3d points in vehicle frame.
    # points_all = np.concatenate(points, axis=0)
    # points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # # point labels.
    # point_labels_all = np.concatenate(point_labels, axis=0)
    # point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # # camera projection corresponding to each point.
    # cp_points_all = np.concatenate(cp_points, axis=0)
    # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    # data_dict['points_all'] = points_all
    # data_dict['point_labels_all'] = point_labels_all
    # data_dict['cp_points_all'] = cp_points_all

    return data_dict

def extract_onesegment_todicts(fileidx, tfrecord_pathnames, step, save_folder, prefix):
    out_dir = Path(save_folder)

    segment_path = tfrecord_pathnames[fileidx]
    c_start = time.time()
    print(
        f'extracting {fileidx}, path: {segment_path}, currenttime: {c_start}')

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
    # framesdict = {}  # []
    Final_array=[]
    for i, data in enumerate(dataset):
        if i % step != 0:  # Downsample
            continue

        # print('.', end='', flush=True) #progress bar
        frame = open_dataset.Frame()#dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            #frames.append(frame)
            print("frame_idx:", i)
            context_name = frame.context.name
            print('context_name:', context_name)#14824622621331930560_2395_420_2415_420, same to the tfrecord file name
            
            data_dict=processoneframe_todict(frame)
            data_dict['context_name']=context_name

            frame_timestamp_micros = str(frame.timestamp_micros)
            data_dict['frame_timestamp_micros']=frame_timestamp_micros
            # print(frame_timestamp_micros)
            Final_array.append(data_dict)

    filename=str(prefix)+context_name#+'.npy'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / filename, Final_array)

    del frame
    del data_dict
    del Final_array


if __name__ == "__main__":
    # save validation folders to dict files
    #folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
    #folders = ["validation_0005", "validation_0006", "validation_0007"]
    #folders = ["validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
    #folders = ["testing_0000","testing_0001","testing_0002","testing_0003","testing_0004","testing_0005","testing_0006","testing_0007"]
    #folders = ["training_0000","training_0001","training_0002","training_0003","training_0004"]
    #folders = ["training_0005","training_0006","training_0007","training_0008","training_0009"]
    folders = ["training_0010","training_0011","training_0012","training_0013","training_0014"]
    root_path = "/mnt/DATA10T/Datasets/Waymo132/"
    out_dir = "/mnt/DATA10T/Datasets/Waymo132/Outdicts"
    data_files = [path for x in folders for path in glob(
        os.path.join(root_path, x, "*.tfrecord"))]
    print("totoal number of files:", len(data_files))  # 886
    step = 1
    #prefix='train01234_'
    #prefix='train56789_'
    prefix='train1011121314_'
    for fileidx in range(len(data_files)):
        extract_onesegment_todicts(fileidx, data_files, step, out_dir,prefix)
    print("finished")

    # for key, value in framedict.items():
    #   print(key)
# FRONT_BEAM_INCLINATION
# FRONT_LIDAR_EXTRINSIC
# FRONT_RANGE_IMAGE_FIRST_RETURN
# FRONT_RANGE_IMAGE_SECOND_RETURN
# FRONT_CAM_PROJ_FIRST_RETURN
# FRONT_CAM_PROJ_SECOND_RETURN
# REAR_BEAM_INCLINATION
# REAR_LIDAR_EXTRINSIC
# REAR_RANGE_IMAGE_FIRST_RETURN
# REAR_RANGE_IMAGE_SECOND_RETURN
# REAR_CAM_PROJ_FIRST_RETURN
# REAR_CAM_PROJ_SECOND_RETURN
# SIDE_LEFT_BEAM_INCLINATION
# SIDE_LEFT_LIDAR_EXTRINSIC
# SIDE_LEFT_RANGE_IMAGE_FIRST_RETURN
# SIDE_LEFT_RANGE_IMAGE_SECOND_RETURN
# SIDE_LEFT_CAM_PROJ_FIRST_RETURN
# SIDE_LEFT_CAM_PROJ_SECOND_RETURN
# SIDE_RIGHT_BEAM_INCLINATION
# SIDE_RIGHT_LIDAR_EXTRINSIC
# SIDE_RIGHT_RANGE_IMAGE_FIRST_RETURN
# SIDE_RIGHT_RANGE_IMAGE_SECOND_RETURN
# SIDE_RIGHT_CAM_PROJ_FIRST_RETURN
# SIDE_RIGHT_CAM_PROJ_SECOND_RETURN
# TOP_BEAM_INCLINATION
# TOP_LIDAR_EXTRINSIC
# TOP_RANGE_IMAGE_FIRST_RETURN #HxWx6 float32 array with the range image of the first return for this lidar. The six channels are range, intensity, elongation, x, y, and z. The x, y, and z values are in vehicle frame.
# TOP_RANGE_IMAGE_SECOND_RETURN
# TOP_CAM_PROJ_FIRST_RETURN
# TOP_CAM_PROJ_SECOND_RETURN
# FRONT_IMAGE
# FRONT_LEFT_IMAGE
# SIDE_LEFT_IMAGE
# FRONT_RIGHT_IMAGE
# SIDE_RIGHT_IMAGE
# FRONT_INTRINSIC
# FRONT_EXTRINSIC
# FRONT_WIDTH
# FRONT_HEIGHT
# FRONT_LEFT_INTRINSIC
# FRONT_LEFT_EXTRINSIC
# FRONT_LEFT_WIDTH
# FRONT_LEFT_HEIGHT
# FRONT_RIGHT_INTRINSIC
# FRONT_RIGHT_EXTRINSIC
# FRONT_RIGHT_WIDTH
# FRONT_RIGHT_HEIGHT
# SIDE_LEFT_INTRINSIC
# SIDE_LEFT_EXTRINSIC
# SIDE_LEFT_WIDTH
# SIDE_LEFT_HEIGHT
# SIDE_RIGHT_INTRINSIC
# SIDE_RIGHT_EXTRINSIC
# SIDE_RIGHT_WIDTH
# SIDE_RIGHT_HEIGHT
# TOP_RANGE_IMAGE_POSE
# POSE