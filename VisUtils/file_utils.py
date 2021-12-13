""" Helper methods for loading and parsing Waymo KITTI format data.

"""
from __future__ import print_function

import numpy as np
import cv2
import os, math
from scipy.optimize import leastsq
from PIL import Image
import os.path as path

class Object2d(object):
    """ 2d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")

        # extract label, truncation, occlusion
        self.img_name = int(data[0])  # 'Car', 'Pedestrian', ...
        self.typeid = int(data[1])  # truncated pixel ratio [0..1]
        self.prob = float(data[2])
        self.box2d = np.array([int(data[3]), int(data[4]), int(data[5]), int(data[6])])

    def print_object(self):
        print(
            "img_name, typeid, prob: %s, %d, %f"
            % (self.img_name, self.typeid, self.prob)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %d, %d, %d, %d"
            % (self.box2d[0], self.box2d[1], self.box2d[2], self.box2d[3])
        )


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
        pts_3d_rect = self.cart2hom(pts_3d_rect)
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




def read_label(label_filename):
    if path.exists(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects
    else:
        return []


def load_image(img_filename):
    img = cv2.imread(img_filename)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb
    #return cv2.imread(img_filename)

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

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
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

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
    
def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])