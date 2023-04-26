import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (25, 14)
import mayavi.mlab as mlab

from VisUtils.mayavivisualize_utils import draw_lidar, visualize_pts, draw_gt_boxes3d
from file_utils import load_image, read_label, load_velo_scan, WaymoCalibration, compute_box_3d

def main():
    #Basepath=r'.\Waymo\sampledata' #r'..\Waymo\sampledata' #r'D:\\Dataset\\WaymoKittitraining_0000\\'+'velodyne'
    Basepath='./Kitti/sampledata'#r'.\Kitti\sampledata'
    idx=8#30
    lidar_filename = os.path.join(Basepath, "velodyne", "%06d.bin" % (idx))
    print(lidar_filename)

    dtype=np.float32
    n_vec=4
    scan = np.fromfile(lidar_filename, dtype=dtype)
    pc_velo = scan.reshape((-1, n_vec))
    pc_velo.shape#(168905, 4) 
    #Each point encodes XYZ + reflectance in Velodyne coordinate: x = forward, y = left, z = up

    #Filter Lidar Points
    point_cloud_range=[0, -15, -5, 90, 15, 4]#[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    mask = (pc_velo[:, 0] >= point_cloud_range[0]) & (pc_velo[:, 0] <= point_cloud_range[3]) \
           & (pc_velo[:, 1] >= point_cloud_range[1]) & (pc_velo[:, 1] <= point_cloud_range[4]) \
           & (pc_velo[:, 2] >= point_cloud_range[2]) & (pc_velo[:, 2] <= point_cloud_range[5]) \
           & (pc_velo[:, 3] <= 1) 
    filteredpoints=pc_velo[mask] #(43376, 4)
    print(filteredpoints.shape)

    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(filteredpoints, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=True, drawregion=True, point_cloud_range=point_cloud_range)
    #visualize_pts(filteredpoints, fig=fig, show_intensity=True)
    mlab.show()
    # input_str = raw_input()
    #mlab.clf()

INSTANCE3D_Color = {
    'Car':(0, 1, 0), 'Pedestrian':(0, 1, 1), 'Sign': (1, 1, 0), 'Cyclist':(0.5, 0.5, 0.3)
}#'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'

def readwaymo():
    Basepath='./Waymo/sampledata' #r'..\Waymo\sampledata' #r'D:\\Dataset\\WaymoKittitraining_0000\\'+'velodyne'
    #Basepath=r'.\Kitti\sampledata'
    data_idx=30

    filename="%06d.png" % (data_idx)
    image_file = os.path.join(Basepath, 'image_0', filename)
    calibration_file = os.path.join(Basepath, 'calib', filename.replace('png', 'txt'))
    label_file = os.path.join(Basepath, 'label_all', filename.replace('png', 'txt')) #'label_0'
    lidar_filename = os.path.join(Basepath, 'velodyne', filename.replace('png', 'bin'))

    object3dlabel=read_label(label_file)
    #print(object3dlabel)
    box=object3dlabel[0]
    print(box)
    data=[box.t[0], box.t[1], box.t[2], box.l, box.w, box.h, box.ry, box.type]#x, y,z
    print(data)
    print(box.box2d) #'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax'

    rgb=load_image(image_file)
    img_height, img_width, img_channel = rgb.shape
    plt.imshow(rgb)
    print(data_idx, "image shape: ", rgb.shape)


    calib=WaymoCalibration(calibration_file)

    n_vec = 4
    pc_veloall=load_velo_scan(lidar_filename, np.float32, n_vec)
    pc_velo = pc_veloall[:, 0:4]
    print(data_idx, "velo  shape: ", pc_velo.shape)

    #Filter Lidar Points
    point_cloud_range=[-50, -15, -5, 90, 15, 4]#[0, -39.68, -3, 69.12, 39.68, 1] # 0:xmin, 1: ymin, 2: zmin, 3: xmax, 4: ymax, 5: zmax
    mask = (pc_velo[:, 0] >= point_cloud_range[0]) & (pc_velo[:, 0] <= point_cloud_range[3]) \
           & (pc_velo[:, 1] >= point_cloud_range[1]) & (pc_velo[:, 1] <= point_cloud_range[4]) \
           & (pc_velo[:, 2] >= point_cloud_range[2]) & (pc_velo[:, 2] <= point_cloud_range[5]) \
           & (pc_velo[:, 3] <= 1) 
    filteredpoints=pc_velo[mask] #(43376, 4)
    print(filteredpoints.shape)
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    draw_lidar(filteredpoints, fig=fig, pts_scale=5, pc_label=False, color_by_intensity=True, drawregion=True, point_cloud_range=point_cloud_range)
    #visualize_pts(filteredpoints, fig=fig, show_intensity=True)

    #only draw camera 0's 3D label
    camera_index=0
    ref_cameraid=0 #3D labels are annotated in camera 0 frame
    #object3dlabel is read in the previous cell
    #object3dlabel=objectlabels[camera_index]#camera 0
    color = (0, 1, 0)
    for obj in object3dlabel:
        if obj.type == "DontCare":
            continue
        print(obj.type)
        # Draw 3d bounding box
        _, box3d_pts_3d = compute_box_3d(obj, calib.P[camera_index])
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d, ref_cameraid)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)
        #draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
        colorlabel=INSTANCE3D_Color[obj.type]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=colorlabel, label=obj.type)


    mlab.show()

if __name__ == '__main__':
    main()
    #readwaymo()