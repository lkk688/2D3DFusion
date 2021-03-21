"""
# * Basic Information
Reads the HDFStore
"frame_{counter}"
"box_metadata_{counter}_{box_counter}"

# * Usage
- `output_filename`: Reads the .h5 file that has been created from `numpy_extract_merged_pointclouds.py`
"""
import numpy as np
import pandas as pd
import open3d as o3d

import logging
logging.basicConfig(level=logging.DEBUG)


def count_frame(store):
    start = 0
    while True:
        try:
            start_frame = f"frame_{start}"
            _ = store[start_frame]
            start += 1
        except:
            break
    return (start - 1)


def get_all_boxes(store, counter):
    box_counter = 0
    boxes = list()
    while True:
        try:
            current_box = f"box_metadata_{counter}_{box_counter}"
            boxes.append(store[current_box])
            box_counter += 1
        except:
            break
    return boxes


def convert_bounding_box_to_points(cx, cy, cz, length, width, height, heading):
    """
    # https://github.com/waymo-research/waymo-open-dataset/issues/108
    """
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


def convert_bounding_box_type_to_color(type):
    """
    1: TYPE_VEHICLE
    2: TYPE_PEDESTRIAN
    3: TYPE_SIGN
    4: CYCLIST
    """
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4

    # COLORS
    YELLOW = (230/255, 230/255, 0)
    GREEN = (0, 255/255, 0)
    RED = (255/255, 0, 0)
    BLUE = (0, 0, 255/255)
    type = int(type)
    rcolor = BLUE
    if type == TYPE_VEHICLE:
        rcolor = BLUE
    elif type == TYPE_PEDESTRIAN:
        rcolor = YELLOW
    elif type == TYPE_SIGN:
        rcolor = RED
    elif type == TYPE_CYCLIST:
        rcolor = GREEN
    else:
        logging.critical("Color not found")
    return rcolor


output_filename = "data/numpy_processed.h5"
# output_filename = "numpy_processed.h5"
store = pd.HDFStore(output_filename, complevel=0)

current_max = count_frame(store)
logging.info(f"Max number of frames in file: {current_max}")

# Initialize the display
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1080, height=720)

# * Store the point cloud here for updating
pcd = o3d.geometry.PointCloud()

# * Store different bounding boxes here
bounding_boxes = list()

# * Get the current frame counter from this variable
counter = 0


def update_frame():
    """
    - Updated `pcd` -> point cloud data
    - Removes previous bounding boxes (previous frame)
    - Adds new bounding boxes (current frame)
    """
    global boxes
    dataframe = store[f'frame_{counter}']
    pcd.points = o3d.utility.Vector3dVector(dataframe.to_numpy())
    # pcd.paint_uniform_color([0, 0, 0])  # Black color
    boxes = get_all_boxes(store, counter)

    # Remove the previous bounding boxes
    for b in bounding_boxes:
        vis.remove_geometry(b, reset_bounding_box=False)

    # Create the Bounding box
    # Add color to the Bounding boxes
    for b in boxes:
        box_points = convert_bounding_box_to_points(
            b[0], b[1], b[2], b[3], b[4], b[5], -b[6])
        type = b[7]
        bounding_box = o3d.geometry.OrientedBoundingBox().create_from_points(
            o3d.utility.Vector3dVector(box_points))
        bounding_box.color = convert_bounding_box_type_to_color(type)
        bounding_boxes.append(bounding_box)
        vis.add_geometry(bounding_box, reset_bounding_box=False)


def display_next_frame(event=None):
    global counter
    logging.info(f"Currently: {counter}")
    if counter >= current_max:
        return

    counter += 1
    logging.info(f"Moved to {counter}\n-----")
    update_frame()


def display_previous_frame(event=None):
    global counter
    logging.info(f"Currently: {counter}")
    if counter <= 0:
        return

    counter -= 1
    logging.info(f"Moved to {counter}\n-----")
    update_frame()


# These functions increment and decrement the current frame counter
vis.register_key_callback(262, display_next_frame)  # Right arrow key
vis.register_key_callback(263, display_previous_frame)  # left arrow key

# Display the data
update_frame()
vis.add_geometry(pcd)
while True:
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
store.close()
