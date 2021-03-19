# 3DDepth

## Kitti data format
Kitti data collection platform [link](http://www.cvlibs.net/datasets/kitti/setup.php):
![image](https://user-images.githubusercontent.com/6676586/111712193-5e6a3000-880a-11eb-8218-cabd7c22862d.png)

The sensor coordinate is shown in the following figure:
![image](https://user-images.githubusercontent.com/6676586/111712395-c6207b00-880a-11eb-9f7e-93cfcbe64108.png)

<img src="https://user-images.githubusercontent.com/6676586/111712591-29121200-880b-11eb-85e7-baf2692d42d0.png" width=300 align=right>

The coordinate system of the following sensors:
* Camera: x = right, y = down, z = forward
* Velodyne: x = forward, y = left, z = up
* GPS/IMU: x = forward, y = left, z = up

The cameras are mounted approximately level with the ground plane. The camera images are cropped to a size of 1382 x 512 pixels using libdc's format 7 mode. After rectification, the images get slightly smaller. The cameras are triggered at 10 frames per second by the laser scanner (when facing forward) with shutter time adjusted dynamically (maximum shutter time: 2 ms). 


The laser scanner spins at 10 frames per second, capturing approximately 100k points per cycle. The vertical resolution of the laser scanner is 64.  Lidar point cloud is stored in fileid.bin: 2D array with shape [num_points, 4] Each point encodes XYZ + reflectance.

Object instance is stored in fileid_label.txt : For each row, the annotation of each object is provided with 15 columns representing certain metadata and 3D box properties in camera coordinates: type | truncation | visibility | observation angle | xmin | ymin |xmax | ymax | height | width | length | tx | ty | tz | roty
Some instances typeare marked as ‘DontCare’, indicating they are not labelled.

In [Kitti Devkit](https://github.com/bostondiditeam/kitti/tree/master/resources/devkit_object), the full definition is
![image](https://user-images.githubusercontent.com/6676586/111712784-91f98a00-880b-11eb-876d-353af30bbbf6.png)

The difference between rotation_y (Rotation ry around Y-axis in camera coordinates [-pi..pi]) and alpha (Observation angle of object, ranging [-pi..pi]) is, that rotation_y is directly given in camera coordinates, while alpha also considers the vector from the camera center to the object center, to compute the relative orientation of the object with respect to the camera. 
* For example, a car which is facing along the X-axis of the camera coordinate system corresponds to rotation_y=0, no matter where it is located in the X/Z plane (bird's eye view)
* alpha is zero when this object is located along the Z-axis (front) of the camera. 


### Calibration
The "calib" folder contains all calibration parameters as shown in the following image. All calibration files are similar, except the first 000000.txt file. The calibration parameters are stored in row-major order. It contains the 3x4 projection matrix parameters which describe the mapping of 3D points in the world to 2D points in an image.
![image](https://user-images.githubusercontent.com/6676586/111711927-eb60b980-8809-11eb-90fb-a5ff6f731639.png)

All matrices are stored row-major, i.e., the first values correspond to the first row. The calibration is done with cam0 as the reference sensor. The laser scanner is registered with respect to the reference camera coordinate system. Rectification R_ref2rect has also been considered during calibration to correct for planar alignment between cameras. The coordinates in the camera coordinate system can be projected in the image by using the 3x4 projection matrix in the calib folder (the left color camera (camera2) should use P2). 

* R0_rect : rotation to account for rectification for points in the reference camera. R0_rect contains a 3x3 matrix which you need to extend to a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere
* Tr_velo_to_cam is 3x4 matrix (R|t): euclidean transformation from lidar to reference camera cam0. You need to extend to a 4x4 matrix in the same way



Projection from lidar to camera 2 project_velo_to_cam2: Suppose we would like to convert Velodyne points into camera coordinate, the following transformations are considered: proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref. The conversion process is shown in the following figure.

![image](https://user-images.githubusercontent.com/6676586/111715035-37aef800-8810-11eb-8938-3a328fdd3ec7.png)







