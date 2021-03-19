# 3DDepth

## Kitti data format
Kitti data collection platform [link](http://www.cvlibs.net/datasets/kitti/setup.php):
![image](https://user-images.githubusercontent.com/6676586/111712193-5e6a3000-880a-11eb-8218-cabd7c22862d.png | width=100)

The laser scanner spins at 10 frames per second, capturing approximately 100k points per cycle. The vertical resolution of the laser scanner is 64. The cameras are mounted approximately level with the ground plane. The camera images are cropped to a size of 1382 x 512 pixels using libdc's format 7 mode. After rectification, the images get slightly smaller. The cameras are triggered at 10 frames per second by the laser scanner (when facing forward) with shutter time adjusted dynamically (maximum shutter time: 2 ms). 

The sensor coordinate is shown in the following figure:
![image](https://user-images.githubusercontent.com/6676586/111712395-c6207b00-880a-11eb-9f7e-93cfcbe64108.png | width=100)

The coordinate system of the following sensors:
* Camera: x = right, y = down, z = forward
* Velodyne: x = forward, y = left, z = up
* GPS/IMU: x = forward, y = left, z = up
![image](https://user-images.githubusercontent.com/6676586/111712591-29121200-880b-11eb-85e7-baf2692d42d0.png | width=100)

Lidar point cloud is stored in fileid.bin: 2D array with shape [num_points, 4] Each point encodes XYZ + reflectance.

Object instance is stored in fileid_label.txt : For each row, the annotation of each object is provided with 15 columns representing certain metadata and 3D box properties in camera coordinates: type | truncation | visibility | observation angle | xmin | ymin |xmax | ymax | height | width | length | tx | ty | tz | roty
Some instances typeare marked as ‘DontCare’, indicating they are not labelled.
In [Kitti Devkit](https://github.com/bostondiditeam/kitti/tree/master/resources/devkit_object), the full definition is
![image](https://user-images.githubusercontent.com/6676586/111712784-91f98a00-880b-11eb-876d-353af30bbbf6.png)






### Calibration
The "calib" folder contains all calibration parameters as shown in the following image. All calibration files are similar, except the first 000000.txt file. The calibration parameters are stored in row-major order. It contains the 3x4 projection matrix parameters which describe the mapping of 3D points in the world to 2D points in an image.
![image](https://user-images.githubusercontent.com/6676586/111711927-eb60b980-8809-11eb-90fb-a5ff6f731639.png)

The calibration is done with cam0 as the reference sensor. The laser scanner is registered with respect to the reference camera coordinate system.



