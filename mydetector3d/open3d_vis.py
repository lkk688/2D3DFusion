
#ref: https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/visualizer/open3d_vis.py
import copy

import numpy as np
import torch

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 mode='xyz'):
    """Draw points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)

    return pcd, points_colors


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.
    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_boxes(points,
                   bbox3d=None,
                   show=True,
                   save_path=None,
                   points_size=2,
                   point_color=(0.5, 0.5, 0.5),
                   bbox_color=(0, 1, 0),
                   points_in_box_color=(1, 0, 0),
                   rot_axis=2,
                   center_mode='lidar_bottom',
                   mode='xyz'):
    """Draw bbox and points on visualizer.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7], optional):
            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            Defaults to None.
        show (bool, optional): whether to show the visualization results.
            Default: True.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is bottom
            center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points, available
            mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes(bbox3d, vis, points_colors, pcd, bbox_color,
                     points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


def _draw_bboxes_ind(bbox3d,
                     vis,
                     indices,
                     points_colors,
                     pcd=None,
                     bbox_color=(0, 1, 0),
                     points_in_box_color=(1, 0, 0),
                     rot_axis=2,
                     center_mode='lidar_bottom',
                     mode='xyz'):
    """Draw bbox on visualizer and change the color or points inside bbox3d
    with indices.
    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        indices (numpy.array | torch.tensor, shape=[N, M]):
            indicate which bbox3d that each point lies in.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        # TODO: fix problem of current coordinate system
        # dim[0], dim[1] = dim[1], dim[0]  # for current coordinate
        # yaw[rot_axis] = -(bbox3d[i, 6] - 0.5 * np.pi)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            points_colors[indices[:, i].astype(np.bool)] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


def show_pts_index_boxes(points,
                         bbox3d=None,
                         show=True,
                         indices=None,
                         save_path=None,
                         points_size=2,
                         point_color=(0.5, 0.5, 0.5),
                         bbox_color=(0, 1, 0),
                         points_in_box_color=(1, 0, 0),
                         rot_axis=2,
                         center_mode='lidar_bottom',
                         mode='xyz'):
    """Draw bbox and points on visualizer with indices that indicate which
    bbox3d that each point lies in.
    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3D bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            Defaults to None.
        show (bool, optional): whether to show the visualization results.
            Default: True.
        indices (numpy.array | torch.tensor, shape=[N, M], optional):
            indicate which bbox3d that each point lies in. Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    # TODO: support score and class info
    assert 0 <= rot_axis <= 2

    # init visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)

    # draw points
    pcd, points_colors = _draw_points(points, vis, points_size, point_color,
                                      mode)

    # draw boxes
    if bbox3d is not None:
        _draw_bboxes_ind(bbox3d, vis, indices, points_colors, pcd, bbox_color,
                         points_in_box_color, rot_axis, center_mode, mode)

    if show:
        vis.run()

    if save_path is not None:
        vis.capture_screen_image(save_path)

    vis.destroy_window()


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.
    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str, optional): path to save visualized results.
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path=None,
                 points_size=2,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.
        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, x_size, y_size, z_size, yaw)
                to be visualized. The 3d bbox is in mode of
                Box3DMode.DEPTH with gravity_center (please refer to
                core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Default: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Default: None.
        """
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def add_seg_mask(self, seg_mask_colors):
        """Add segmentation mask to visualizer via per-point colorization.
        Args:
            seg_mask_colors (numpy.array, shape=[N, 6]):
                The segmentation mask whose first 3 dims are point coordinates
                and last 3 dims are converted colors.
        """
        # we can't draw the colors on existing points
        # in case gt and pred mask would overlap
        # instead we set a large offset along x-axis for each seg mask
        self.seg_num += 1
        offset = (np.array(self.pcd.points).max(0) -
                  np.array(self.pcd.points).min(0))[0] * 1.2 * self.seg_num
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[offset, 0, 0])  # create coordinate frame for seg
        self.o3d_visualizer.add_geometry(mesh_frame)
        seg_points = copy.deepcopy(seg_mask_colors)
        seg_points[:, 0] += offset
        _draw_points(
            seg_points, self.o3d_visualizer, self.points_size, mode='xyzrgb')

    def show(self, save_path=None):
        """Visualize the points cloud.
        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return

import argparse
import os

def testmmdetection3dresult(resultpath='./outputs/lidarresultsnp.npy'):
    lidarresults=np.load(resultpath,allow_pickle=True)
    #print("npy results length:", len(lidarresults))
    lidarresult=lidarresults.item()
    points = lidarresult['points']#x,4 shape, boxes_3d, labels_3d
    pred_bboxes = lidarresult['boxes_3d']
    return points, pred_bboxes


#-------new add
class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_diffculty(self):
        """ Function that estimate difficulty to detect the object as defined in kitti website"""
        # height of the bounding box
        bb_height = np.abs(self.xmax - self.xmin)

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif (
            bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50
        ):
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty of estimation: {}".format(self.estimate_diffculty()))


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    new3dboxes=[]
    for object in objects:
        #3D bbox (x, y, z, x_size, y_size, z_size, yaw)
        new3dbox = [object.t[0], object.t[1], object.t[2], object.t[0],object.h,object.w,object.l,object.ry]
        new3dboxes.append(new3dbox)
    return objects, new3dboxes

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default='/mnt/DATA10T/Datasets/Kitti/training/', help="root folder"
    )
    parser.add_argument(
        "--index", default="1", help="Kitti file index"
    )
    parser.add_argument(
        "--result_path", default="0000000000.bin", help="Kitti lidar file"
    )
    flags = parser.parse_args()

    basedir = flags.root_path
    idx = int(flags.index)


    """Load and parse a velodyne binary file."""
    lidar_dir = os.path.join(basedir, "velodyne")
    lidar_filename = os.path.join(lidar_dir, "%06d.bin" % (idx))
    scan = np.fromfile(lidar_filename, dtype=np.float32) # vector points
    points_with_intensity=scan.reshape((-1, 4))#4 points a group
    points = points_with_intensity[:, :3]#115384, 3

    label_dir = os.path.join(basedir, "label_2")
    label_filename = os.path.join(label_dir, "%06d.txt" % (idx))
    objects, new3dboxes=read_label(label_filename) #Object3d list
    #bbox3d (numpy.array, shape=[M, 7]):
    #            3D bbox (x, y, z, x_size, y_size, z_size, yaw)

    
    points, pred_bboxes = testmmdetection3dresult()
    filename='test'
    snapshot=True
    vis = Visualizer(points)
    if pred_bboxes is not None:
        vis.add_bboxes(bbox3d=pred_bboxes)
    show_path = os.path.join(flags.result_path,
                             f'{filename}_online.png') if snapshot else None
    vis.show(show_path)
