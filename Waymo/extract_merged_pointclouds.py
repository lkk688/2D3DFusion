# adapted from 'example/visualize_pcl.py'
"""
# * Basic Information
Creates a HDFStore which contains data in this form
"frame_{counter}" -> each frame data
"box_metadata_{counter}_{box_counter}" -> each frame can contain multiple boxes

# * Usage
You can update the following variables to customize the script
- `filename`: Input waymo .tfrecord file
- `output_file`: The output .h5 file created after parsing .tfrecord file
- `PARSE_NUM_FRAMES`: Number of frames to be parsed (None means all points will be parsed)
- `COMPRESSION_LEVEL`: Compression Level used to store data in h5 file. No noticeable difference between compression level 0-9
- `DEBUG_OUTPUT`: Debug the first frame to check visualizer (NOTE, Automatically goes to the next frame after 10 seconds)
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils

# Add this towards the end of the includes
import logging
logging.basicConfig(level=logging.INFO)


def get_all_lidar_point_data_from_frame(frame, point_cloud_list):
    """
    Get Lidar Point data from all the different angles
    (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR)
    """
    num_points = 0
    for laser_id in range(1, 6):
        laser_name = dataset_pb2.LaserName.Name.DESCRIPTOR.values_by_number[laser_id].name

        # Get the laser information
        laser = utils.get(frame.lasers, laser_id)
        laser_calibration = utils.get(
            frame.context.laser_calibrations, laser_id)

        # Parse the top laser range image and get the associated projection.
        ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(
            laser)

        # Convert the range image to a point cloud.
        pcl, pcl_attr = utils.project_to_pointcloud(
            frame, ri, camera_projection, range_image_pose, laser_calibration)

        # Add all the point clouds to the list
        point_cloud_list.append(pcl)
        logging.debug(f'{laser_name} LiDAR measured {len(pcl)} points.')
        num_points += len(pcl)
    return num_points


def merge_all_lidar_point_data(merged_point_cloud, point_cloud_list):
    shift_index = 0
    for point_cloud in point_cloud_list:
        for idx, point in enumerate(point_cloud):
            merged_point_cloud[shift_index + idx] = point
        shift_index += len(point_cloud)
    return shift_index


def convert_bounding_box_to_points(box):
    """
    # https://github.com/waymo-research/waymo-open-dataset/issues/108
    """
    heading = -box.heading
    cx = box.center_x
    cy = box.center_y
    cz = box.center_z
    length = box.length
    width = box.width
    height = box.height

    corner = np.array([[-0.5 * length, -0.5 * width],
                       [-0.5 * length, 0.5 * width],
                       [0.5 * length, -0.5 * width],
                       [0.5 * length, 0.5 * width]])

    rotation_matrix = np.array([[np.cos(heading), - np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])

    corner = np.matmul(corner, rotation_matrix)

    z_bottom = cz - height / 2
    z_top = cz + height / 2
    p0 = [cx + corner[0][0], cy + corner[0][1], z_bottom]
    p1 = [cx + corner[1][0], cy + corner[1][1], z_bottom]
    p2 = [cx + corner[2][0], cy + corner[2][1], z_bottom]
    p3 = [cx + corner[3][0], cy + corner[3][1], z_bottom]
    p4 = [cx + corner[0][0], cy + corner[0][1], z_top]
    p5 = [cx + corner[1][0], cy + corner[1][1], z_top]
    p6 = [cx + corner[2][0], cy + corner[2][1], z_top]
    p7 = [cx + corner[3][0], cy + corner[3][1], z_top]
    return np.array([p0, p1, p2, p3, p4, p5, p6, p7])


"""
* MAIN STARTS HERE
"""

# Configurable options
# filename = '/data/cmpe249-f20/Waymo/training_0000/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
filename = 'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
output_filename = "numpy_processed.h5"

# * CONSTANTS
PARSE_NUM_FRAMES = 5  # None means it will parse the entire file
COMPRESSION_LEVEL = 0  # Compression 9 makes no difference
DEBUG_OUTPUT = False

# * STATES
datafile = WaymoDataFileReader(filename)
store = pd.HDFStore(output_filename, complevel=COMPRESSION_LEVEL)

table = datafile.get_record_table()
frames = iter(datafile)
frame_number = 0
logging.info(f"There are {len(table)} frames")

counter = 0
for frame in tqdm(frames):
    point_cloud_list = list()
    num_points = get_all_lidar_point_data_from_frame(frame, point_cloud_list)

    num_features = 3
    merged_point_cloud = np.empty((num_points, num_features))
    shift_index = merge_all_lidar_point_data(
        merged_point_cloud, point_cloud_list)

    # Both of these values should be the same
    logging.debug(f"Shift Index: {shift_index} Num Points: {num_points}")
    logging.debug(
        f"Merged_point_cloud: {len(merged_point_cloud)} {type(merged_point_cloud)}")
    dataframe = pd.DataFrame(data=merged_point_cloud,
                             columns=["x", "y", "z"])
    logging.debug(f"Storing Frame {counter}")
    store[f'frame_{counter}'] = dataframe

    box_counter = 0
    # TODO, Try to color code this data
    # ! TYPE_VEHICLE TYPE_SIGN TYPE_PEDESTRIAN  TYPE_CYCLISTS
    for label in frame.laser_labels:
        points = convert_bounding_box_to_points(label.box)
        box_dataframe = pd.DataFrame(data=points, columns=["x", "y", "z"])
        box = label.box
        store[f'box_metadata_{counter}_{box_counter}'] = pd.Series(
            data=[box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading, label.type])
        box_counter += 1
    counter += 1

    if PARSE_NUM_FRAMES != None:
        if counter == PARSE_NUM_FRAMES:
            break

    if DEBUG_OUTPUT == True:
        import open3d as o3d
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=1080, height=720)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dataframe.to_numpy())
        vis.add_geometry(pcd)

        # * Add the bounding box geometry here
        for k in frame.laser_labels:
            points = convert_bounding_box_to_points(k.box)
            bounding_box = o3d.geometry.OrientedBoundingBox(
            ).create_from_points(o3d.utility.Vector3dVector(points))
            vis.add_geometry(bounding_box)

        # Start the render loop here
        start = datetime.datetime.now()
        while True:
            vis.poll_events()
            vis.update_renderer()
            current = datetime.datetime.now()
            if (current - start) > datetime.timedelta(seconds=10.0):
                break

store.close()
