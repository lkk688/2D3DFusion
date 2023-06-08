import os
import json
import argparse
import numpy as np
from pypcd import pypcd
import open3d as o3d
import matplotlib
import json
from tqdm import tqdm
import math

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PIL import Image


def read_image(path_img, img_plot=True):
    img = np.asarray(Image.open(path_img)) #use Pillow to open an image (with PIL.Image.open), and immediately convert the PIL.Image.Image object into an 8-bit (dtype=uint8) numpy array.
    if img_plot == True:
        plt.figure(figsize=(10,6))
        imgplot = plt.imshow(img)
        # lum_img = img[:, :, 0]
        # plt.hist(lum_img.ravel(), bins=range(256), fc='k', ec='k')
    return img

def read_label_bboxes(label_path):
    with open(label_path, "r") as load_f:
        labels = json.load(load_f)

    boxes = []
    for label in labels:
        obj_size = [
            float(label["3d_dimensions"]["l"]),
            float(label["3d_dimensions"]["w"]),
            float(label["3d_dimensions"]["h"]),
        ]
        yaw_lidar = float(label["rotation"])
        center_lidar = [
            float(label["3d_location"]["x"]),
            float(label["3d_location"]["y"]),
            float(label["3d_location"]["z"]),
        ]

        box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)
        boxes.append(box)

    return boxes

def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def read_pcd(path_pcd):
    pointpillar = o3d.io.read_point_cloud(path_pcd)
    points = np.asarray(pointpillar.points)
    return points

def read_pointsbin(path_bin):
    dtype=np.float32
    n_vec=4
    scan = np.fromfile(path_bin, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan

#https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_colortable(colors, heightlevel, ncols=4):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        if i>=len(heightlevel):
            break
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        txtname="height: "+str(heightlevel[i])
        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')
        ax.text(text_pos_x+200, y, txtname, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

def colorlevel(maxlevel, heightlevel):
    #colors = matplotlib.colors.XKCD_COLORS.values()
    colors=mcolors.TABLEAU_COLORS

    max_color_num = min(maxlevel, len(heightlevel))
    plot_colortable(colors, heightlevel, ncols=1)

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)#[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]
    return label_rgba
    #label_rgba = label_rgba.squeeze()[:, :3]


def generatecolor(points):
    heightlevel=[-3,-2,-1,0,1,2,3]
    colors = colorlevel(7, heightlevel) #(7,3)
    heightarray=points[:,2]
    minheight = min(heightarray) #-4.29
    sortedpoint = np.sort(heightarray, axis=0)
    print(minheight)

    colordata=np.ones((points.shape[0], 3)) # range [0, 1]
    for i in range(points.shape[0]):
        #if points.shape[1]==3: #using height as color
        if points[i,2]<heightlevel[0]:
            colordata[i,:] = colors[0,:]
        elif points[i,2]>=heightlevel[0] and points[i,2]<heightlevel[1]:
            colordata[i,:] = colors[1,:]
        elif points[i,2]>=heightlevel[1] and points[i,2]<heightlevel[2]:
            colordata[i,:] = colors[2,:]
        elif points[i,2]>heightlevel[2] and points[i,2]<heightlevel[3]:
            colordata[i,:] = colors[3,:]
        elif points[i,2]>heightlevel[3] and points[i,2]<heightlevel[4]:
            colordata[i,:] = colors[4,:]
        elif points[i,2]>heightlevel[4] and points[i,2]<heightlevel[5]:
            colordata[i,:] = colors[5,:]
        elif points[i,2]>=heightlevel[4]:
            colordata[i,:] = colors[6,:]
            #colordata[i,0]=min(points[i,2],1)
        # elif points.shape[1]==4: #using intensity as color
        #     colordata[i,0]=min(points[i,3],1)
    return colordata

def draw_pcd(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        colordata = generatecolor(points)
        pts.colors = o3d.utility.Vector3dVector(colordata)
    else:
        pts.colors = o3d.utility.Vector3dVector(point_colors) #[1, 0, 0] is red

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    # if ref_boxes is not None:
    #     vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def concatenate_points(pc1, pc2, path_save):
    print(pc1.shape)
    print(pc2.shape)
    fusedpoints = np.append(pc1, pc2, axis=0)
    print(fusedpoints.shape)
    #fusedpoints.tofile(path_save)

def concatenate_points_i_and_v(path_c, path_i2v, path_dest):

    path_c_data_info = os.path.join(path_c, "cooperative/data_info.json")
    c_data_info = read_json(path_c_data_info)

    for data in tqdm(c_data_info):
        fusedpoints, name_v, name_i= concatenate_datainfo(path_c, path_i2v, data)
        name_v = name_v.replace(".pcd", ".bin")
        path_save = os.path.join(path_dest, name_v)
        fusedpoints.tofile(path_save)
        
def concatenate_datainfo(path_c, path_i2v, data):
    path_pcd_v = os.path.join(path_c, data["vehicle_pointcloud_path"]) #vehicle pointcloud path
    name_v = os.path.split(path_pcd_v)[-1] #vehicle pointcloud file name

    name_i = os.path.split(data["infrastructure_pointcloud_path"])[-1] #infrastructure points name
    path_pcd_i = os.path.join(path_i2v, name_i)

    points_v = read_pcd(path_pcd_v)
    points_i = read_pcd(path_pcd_i)
    print(points_v.shape)
    print(points_i.shape)
    fusedpoints = np.append(points_v, points_i, axis=0)
    print(fusedpoints.shape)
    return fusedpoints, name_v, name_i

Dataset_root='/mnt/f/Dataset/DAIR-C/cooperative-vehicle-infrastructure-example_10906136335224832/cooperative-vehicle-infrastructure-example/'    
parser = argparse.ArgumentParser("Convert The Point Cloud from Infrastructure to Ego-vehicle")
parser.add_argument(
    "--source-root",
    type=str,
    default=Dataset_root,
    help="Raw data root about DAIR-V2X-C.",
)
parser.add_argument(
    "--i2v-root",
    type=str,
    default=Dataset_root+"vic3d-early-fusion/velodyne/lidar_i2v/",
    help="The data root where the data with ego-vehicle coordinate is generated.",
)
parser.add_argument(
    "--target-root",
    type=str,
    default=Dataset_root+"vic3d-early-fusion/velodyne-concated/",
    help="The concated point cloud.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root
    path_i2v = args.i2v_root
    path_v = source_root+'vehicle-side/velodyne/'
    path_i = source_root+'infrastructure-side/velodyne/'

    data_info_path=source_root+'cooperative/data_info.json'
    data_info = read_json(data_info_path)

    example_info=data_info[0]
    infrastructure_image_path=example_info['infrastructure_image_path'] #'infrastructure-side/image/000049.jpg'
    infrastructure_image = read_image(source_root+infrastructure_image_path)
    infrastructure_pointcloud_path=example_info['infrastructure_pointcloud_path'] # 'infrastructure-side/velodyne/000049.pcd'
    infrastructure_pointcloud=read_pcd(source_root+infrastructure_pointcloud_path)
    draw_pcd(infrastructure_pointcloud)
    vehicle_image_path=example_info['vehicle_image_path'] #'vehicle-side/image/015404.jpg'
    vehicle_image  = read_image(source_root+vehicle_image_path)
    vehicle_pointcloud_path=example_info['vehicle_pointcloud_path'] #'vehicle-side/velodyne/015404.pcd'
    vehicle_pointcloud = read_pcd(source_root+vehicle_pointcloud_path)
    draw_pcd(vehicle_pointcloud)
    cooperative_label_path=example_info['cooperative_label_path'] #'cooperative/label_world/015404.json'
    cooperative_label = read_json(source_root+cooperative_label_path)







    # binpoints=read_pointsbin("/home/lkk/Developer/data/v2xvehiclekitti/velodyne/001766.bin")#converted to kitti
    # draw_pcd(binpoints)

    # vehicleLidar_path=path_v+'015344.pcd'
    # vehicle_points = read_pcd(vehicleLidar_path)
    # draw_pcd(vehicle_points) #(63528, 3)

    # infraLidar_path=path_i+'000009.pcd'
    # Infra_points = read_pcd(infraLidar_path)
    # draw_pcd(Infra_points)

    # i2vlidar_path=path_i2v+'000016.pcd'
    # i2vpoints = read_pcd(i2vlidar_path)
    # draw_pcd(i2vpoints)

    path_c_data_info = os.path.join(source_root, "cooperative/data_info.json")
    c_data_info = read_json(path_c_data_info)
    data=c_data_info[0]
    fusedpoints, name_v, name_i = concatenate_datainfo(source_root, path_i2v, data)
    draw_pcd(fusedpoints)

    boxes = read_label_bboxes(cooperative_label_path)
    draw_pcd(fusedpoints, gt_boxes=boxes)


