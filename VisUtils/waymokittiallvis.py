#based on drawkittilidar.py
import numpy as np
import os
import cv2
import sys
import argparse
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (25, 14)
import mayavi.mlab as mlab

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#Current folder
ROOT_DIR = os.path.dirname(BASE_DIR)#Project root folder
sys.path.append(ROOT_DIR)


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


def filter_lidarpoints(pc_velo, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    #Filter Lidar Points
    #point_cloud_range=[0, -15, -5, 90, 15, 4]#[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    mask = (pc_velo[:, 0] >= point_cloud_range[0]) & (pc_velo[:, 0] <= point_cloud_range[3]) \
           & (pc_velo[:, 1] >= point_cloud_range[1]) & (pc_velo[:, 1] <= point_cloud_range[4]) \
           & (pc_velo[:, 2] >= point_cloud_range[2]) & (pc_velo[:, 2] <= point_cloud_range[5]) \
           & (pc_velo[:, 3] <= 1) 
    filteredpoints=pc_velo[mask] #(43376, 4)
    print(filteredpoints.shape)
    return filteredpoints

#from https://github.com/open-mmlab/OpenPCDet/tree/master/tools/visual_utils
def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

#from Kitti.viz_util.py
def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=3, #0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
    drawfov=False,
    drawregion=False,
    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2] #Z height
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        #color = pc[:, 2]
        intensities=pc[:, 3]
        maxintensity=max(intensities)
        max_index = np.argmax(intensities, axis=0)
        print(intensities[max_index])
        print(pc[max_index,:])
        minintensity=min(intensities)
        color=np.sqrt(intensities)*10#(intensities-minintensity)

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    #plot3d: Draws lines between points, the positions of the successive points of the line
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),#red, X (0,0,0)->(2,0,0)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[0, 0], axes[0, 1], axes[0, 2], "X", scale=(0.1, 0.1, 0.1)) #(2,0,0) position

    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),#green green, Y (0,2,0)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[1, 0], axes[1, 1], axes[1, 2], "Y", scale=(0.1, 0.1, 0.1)) #(0,2,0) position

    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),#blue Z (0,0,2)
        tube_radius=None,
        figure=fig,
    )
    mlab.text3d(axes[2, 0], axes[2, 1], axes[2, 2], "Z", scale=(0.1, 0.1, 0.1)) #(0,0,2) position

    if drawfov:
        # draw fov (todo: update to real sensor spec.)
        fov = np.array(
            [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
        )

        mlab.plot3d(
            [0, fov[0, 0]],
            [0, fov[0, 1]],
            [0, fov[0, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
        mlab.plot3d(
            [0, fov[1, 0]],
            [0, fov[1, 1]],
            [0, fov[1, 2]],
            color=(1, 1, 1),
            tube_radius=None,
            line_width=1,
            figure=fig,
        )

    if drawregion:
        #point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
        # draw square region
        x1 = point_cloud_range[0]#TOP_X_MIN
        x2 = point_cloud_range[3]#TOP_X_MAX
        y1 = point_cloud_range[1]#TOP_Y_MIN
        y2 = point_cloud_range[4]#TOP_Y_MAX
        linewidth=0.2
        tuberadius=0.01 #0.1
        mlab.plot3d(
            [x1, x1],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x2, x2],
            [y1, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y1, y1],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )
        mlab.plot3d(
            [x1, x2],
            [y2, y2],
            [0, 0],
            color=(0.5, 0.5, 0.5),
            tube_radius=tuberadius,
            line_width=linewidth,
            figure=fig,
        )

    # mlab.orientation_axes()
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig

def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def read_label(label_filename):
    if os.path.exists(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects
    else:
        return []

def read_multi_label(label_files):
    objectlabels=[]
    for label_file in label_files:
        object3dlabel=read_label(label_file)
        objectlabels.append(object3dlabel)
    return objectlabels
import matplotlib.pyplot as plt
import matplotlib.patches as patches

INSTANCE_Color = {
    'Car':'red', 'Pedestrian':'green', 'Sign': 'yellow', 'Cyclist':'purple'
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'

cameraorder={
        0:1, 1:0, 2:2, 3:3, 4:4
    }#Front, front_left, side_left, front_right, side_right
cameraname_map={0:"FRONT", 1:"FRONT_LEFT", 2:"FRONT_RIGHT", 3:"SIDE_LEFT", 4:"SIDE_RIGHT"}

def plt_multiimages(images, objectlabels, order=1):
    plt.figure(order, figsize=(25, 20))
    camera_count = len(images)
    for count in range(camera_count):#each frame has 5 images
        index=cameraorder[count]
        pltshow_image_with_boxes(index, images[index], objectlabels[index], [3, 3, count+1])

def pltshow_image_with_boxes(cameraid, img, objects, layout, cmap=None):
    ax = plt.subplot(*layout)
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    plt.imshow(img1, cmap=cmap)
    plt.title(cameraname_map[cameraid])
    
    if not objects or len(objects)==0: #no objects
        return
    for obj in objects:
        if obj.type == "DontCare":
            continue
        box=obj.box2d
        objectclass=obj.type
        colorlabel=INSTANCE_Color[objectclass]
        [xmin, ymin, xmax, ymax]=box
        width=xmax-xmin #box.length
        height=ymax-ymin #box.width
        if (height>0 and width>0):
            print(box)
#             xmin=label.box.center_x - 0.5 * label.box.length
#             ymin=label.box.center_y - 0.5 * label.box.width
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
                xy=(xmin,ymin),
                width=width, #label.box.length,
                height=height, #label.box.width,
                linewidth=1,
                edgecolor=colorlabel,
                facecolor='none'))
            ax.text(xmin, ymin, objectclass, color=colorlabel, fontsize=8)
    # Show the camera image.
    plt.grid(False)
    plt.axis('on')

INSTANCE3D_ColorCV2 = {
    'Car':(0, 255, 0), 'Pedestrian':(255, 255, 0), 'Sign': (0, 255, 255), 'Cyclist':(127, 127, 64)
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'

def plt3dbox_images(images,objectlabels,calib):
    plt.figure(figsize=(25, 20))
    camera_count = len(images)
    for count in range(camera_count):#each frame has 5 images
        index=cameraorder[count]
        img = images[index]
        object3dlabel=objectlabels[index]
        pltshow_image_with_3Dboxes(index, img, object3dlabel,calib, [3, 3, count+1])

def pltshow_image_with_3Dboxes(cameraid, img, objects, calib, layout, cmap=None):
    ax = plt.subplot(*layout)
    """ Show image with 3D bounding boxes """
    img2 = np.copy(img)  # for 3d bbox
    #plt.figure(figsize=(25, 20))
    print("camera id:", cameraid)
    if cameraid ==0:
        for obj in objects:
            if obj.type == "DontCare" or (obj is None):
                continue
            box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 8x3
            box3d_pts_2d = calib.project_cam3d_to_image(box3d_pts_3d, cameraid) #return (8,2) array in left image coord.
            #print("obj:", box3d_pts_2d)
            if box3d_pts_2d is not None:
                colorlabel=INSTANCE3D_ColorCV2[obj.type]
                img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)
    else:
        ref_cameraid=0
        for obj in objects:
            if obj.type == "DontCare" or (obj is None):
                continue
            #_, box3d_pts_3d = compute_box_3d(obj, calib.P[camera_index]) #get 3D points in label (in camera 0 coordinate), convert to 8 corner points
            box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid) # convert the 3D points to velodyne coordinate
            box3d_pts_3d_cam=calib.project_velo_to_cameraid(box3d_pts_3d_velo,cameraid) # convert from the velodyne coordinate to camera coordinate (cameraid)
            box3d_pts_2d=calib.project_cam3d_to_image(box3d_pts_3d_cam,cameraid) # project 3D points in cameraid coordinate to the imageid coordinate (2D 8 points)
            if box3d_pts_2d is not None:
                print(box3d_pts_2d)
                colorlabel=INSTANCE3D_ColorCV2[obj.type]
                img2 = draw_projected_box3d(img2, box3d_pts_2d, color=colorlabel)

    plt.imshow(img2, cmap=cmap)
    plt.title(cameraname_map[cameraid])
    plt.grid(False)
    plt.axis('on')

def load_image(img_filenames, showfig=True):
    imgs=[]
    for img_filename in img_filenames:
        img = cv2.imread(img_filename)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)
    return imgs
    #return cv2.imread(img_filename)

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4, filterpoints=False, point_cloud_range=[0, -15, -5, 90, 15, 4]):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    if filterpoints:
        scan=filter_lidarpoints(scan, point_cloud_range) #point_cloud_range #0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    return scan

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(obj, dataset='kitti'):
    """ Takes an object3D
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    """
    #x-y-z: front-left-up (waymo) -> x_right-y_down-z_front(kitti)
    # compute rotational matrix around yaw axis (camera coord y pointing to the bottom, thus Yaw axis is rotate y-axis)
    R = roty(obj.ry)

    # 3d bounding box dimensions: x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
    l = obj.l #x
    w = obj.w #z
    h = obj.h #y

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    #print(corners_3d)
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1): #in Kitti, z axis is to the front, if z<0.1 means objs in back of camera
        return np.transpose(corners_3d)
    
    return np.transpose(corners_3d)


class WaymoCalibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord. 
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath):
        self.camera_count = 5
        calibs = self.read_calib_file(calib_filepath)
        print(calibs)
        # Projection matrix from rect camera coord to image2 coord
        #self.P = calibs["P0"]#["P2"]
        #self.P = np.reshape(self.P, [3, 4])
        P_name = ["P"+str(i) for i in range(self.camera_count)] 
        self.P = [np.reshape(calibs[name], [3,4]) for name in P_name] #5 cameras
        
        
        # Rigid transform from Velodyne coord to reference camera coord
        #self.V2C = calibs["Tr_velo_to_cam_0"]#calibs["Tr_velo_to_cam"]
        #self.V2C = np.reshape(self.V2C, [3, 4])
        
        Tr_velo_to_cam_name = ["Tr_velo_to_cam_"+str(i) for i in range(self.camera_count)] 
        self.V2C = [np.reshape(calibs[name],[3,4]) for name in Tr_velo_to_cam_name] #array for 5 cameras
        
        #self.C2V = self.inverse_rigid_trans(self.V2C)
        self.C2V = [self.inverse_rigid_trans(self.V2C[i]) for i in range(self.camera_count)]
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"], #'R0_rect': array([1., 0., 0., 0., 1., 0., 0., 0., 1.]), not used in Waymo
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
#         self.c_u = self.P[0, 2]
#         self.c_v = self.P[1, 2]
#         self.f_u = self.P[0, 0]
#         self.f_v = self.P[1, 1]
#         self.b_x = self.P[0, 3] / (-self.f_u)  # relative
#         self.b_y = self.P[1, 3] / (-self.f_v)

    def inverse_rigid_trans(self,Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    # convert 3Dbox in rect camera coordinate to velodyne coordinate
    def project_rect_to_velo(self, pts_3d_rect, camera_id):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)# using R0 to convert to camera rectified coordinate (same to camera coordinate)
        return self.project_ref_to_velo(pts_3d_ref, camera_id)#using C2V
    
    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_velo(self, pts_3d_ref, camera_id):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V[camera_id]))
    
    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
    
    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_cam3d_to_image(self, pts_3d_rect, cameraid):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)#nx3 to nx4 by pending 1
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P[cameraid]))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]
    
    #Liar project to image
    def project_velo_to_cameraid_rect(self, pts_3d_velo, cameraid):
        pts_3d_camid = self.project_velo_to_cameraid(pts_3d_velo, cameraid)
        return self.project_ref_to_rect(pts_3d_camid)#apply R0
    
    def project_velo_to_cameraid(self, pts_3d_velo, cameraid):#project velodyne to camid frame
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C[cameraid]))
    
    def project_velo_to_image(self, pts_3d_velo, cameraid):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_cameraid_rect(pts_3d_velo, cameraid)
        return self.project_cam3d_to_image(pts_3d_rect, cameraid)
    
    def project_3dcornerboxes_to_image(self,corners_3d, cameraid):
        """ project the 3d bounding box into the image plane.

        input: pts_3d: nx3 matrix
                P:      3x4 projection matrix
        output: pts_2d: nx2 matrix

        P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
        => normalize projected_pts_2d(2xn)

        <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
        """
        # 
        corners_2d_test = project_to_image(np.transpose(corners_3d), self.P[cameraid])
        pts_3d = np.transpose(corners_3d) #nx3 matrix
        corners_2d = self.project_cam3d_to_image(pts_3d, cameraid)#P matrix for cameraid
        return corners_2d

def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

INSTANCE3D_Color = {
    'Car':(0, 1, 0), 'Pedestrian':(0, 1, 1), 'Sign': (1, 1, 0), 'Cyclist':(0.5, 0.5, 0.3)
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default=r'D:\Dataset\WaymoKittitraining_0000', help="root folder"
    )#'/mnt/DATA10T/Datasets/Kitti/training/' r'.\Kitti\sampledata' 
    parser.add_argument(
        "--index", default="2079", help="file index"
    )
    parser.add_argument(
        "--dataset", default="kitti", help="file index"
    )
    parser.add_argument(
        "--camera_count", default=5, help="file index"
    )
    flags = parser.parse_args()

    basedir = flags.root_path
    idx = int(flags.index)
    camera_count=flags.camera_count


    """Load and parse a velodyne binary file."""
    camera_index=0
    filename="%06d.png" % (idx)
    image_folder='image_'+str(camera_index)
    #image_file = os.path.join(basedir, image_folder, filename)
    image_files = [os.path.join(basedir, "image_"+str(i), filename) for i in range(camera_count)] 
    calibration_file = os.path.join(basedir, 'calib', filename.replace('png', 'txt'))
    label_all_file = os.path.join(basedir, 'label_all', filename.replace('png', 'txt')) #'label_0'
    labels_files=[os.path.join(basedir, "label_"+str(i), filename.replace('png', 'txt')) for i in range(camera_count)] 
    lidar_filename = os.path.join(basedir, 'velodyne', filename.replace('png', 'bin'))

    #load Lidar points
    dtype=np.float32
    point_cloud_range=[0, -15, -5, 90, 15, 4] #0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    pc_velo=load_velo_scan(lidar_filename, dtype=np.float32, n_vec=4, filterpoints=False, point_cloud_range=point_cloud_range)
    ##Each point encodes XYZ + reflectance in Velodyne coordinate: x = forward, y = left, z = up

    calib=WaymoCalibration(calibration_file)

    object3dlabels=read_label(label_all_file)
    #print(object3dlabel)
    box=object3dlabels[0]
    print(box)
    data=[box.t[0], box.t[1], box.t[2], box.l, box.w, box.h, box.ry, box.type]#x, y,z
    print(data)
    print(box.box2d) #'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax'

    images=load_image(image_files)
    objectlabels=read_multi_label(labels_files)
    plt_multiimages(images, objectlabels, order=1)
    # cv2rgb = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
    # cv2.imshow("Image", cv2rgb)
    # cv2.waitKey(0)

    plt3dbox_images(images,objectlabels,calib)


    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(pc_velo, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=True, drawregion=True, point_cloud_range=point_cloud_range)
    #visualize_pts(pc_velo, fig=fig, show_intensity=True)

    #only draw camera 0's 3D label
    ref_cameraid=0 #3D labels are annotated in camera 0 frame
    color = (0, 1, 0)
    for obj in object3dlabels:
        if obj.type == "DontCare":
            continue
        print(obj.type)
        # Draw 3d bounding box
        box3d_pts_3d = compute_box_3d(obj) #3d box coordinate=>get 8 points in camera rect, 
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid)
        print("box3d_pts_3d_velo:", box3d_pts_3d_velo)
        #draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
        colorlabel=INSTANCE3D_Color[obj.type]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=colorlabel, label=obj.type)

    # rgb=load_image(image_file)
    # img_height, img_width, img_channel = rgb.shape
    # plt.imshow(rgb)
    # print(data_idx, "image shape: ", rgb.shape)


    mlab.show()
    #draw_lidarpoints(pc_velo, point_cloud_range)

    # label_dir = os.path.join(basedir, "label_2")
    # label_filename = os.path.join(label_dir, "%06d.txt" % (idx))
    # objects, new3dboxes=read_label(label_filename) #Object3d list
    #bbox3d (numpy.array, shape=[M, 7]):
    #            3D bbox (x, y, z, x_size, y_size, z_size, yaw)


