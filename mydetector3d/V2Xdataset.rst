DAIR V2X Data
=============

DAIR-V2X Dependencies
---------------------
Use `DAIR-V2X <https://github.com/AIR-THU/DAIR-V2X/tree/main>`_ to read the Lidar pcd file in cooperative sensing dataset. Install the following required packages
.. code-block:: console

  (mypy310) lkk@Alienware-LKKi7G8:~/Developer$ git clone https://github.com/klintan/pypcd.git
  (mypy310) lkk@Alienware-LKKi7G8:~/Developer/pypcd$ python setup.py install

Create a new folder named "dairv2x" under "mydetector3d/datasets/dairv2x"


DAIR V2X Dataset Process
------------------------
DAIR V2X dataset is saved in '/data/cmpe249-fa22/DAIR-C' folder. Based on 'https://github.com/AIR-THU/DAIR-V2X/blob/main/docs/get_started.md', 
  * 'cooperative-vehicle-infrastructure' folder as the follow three sub-folders: cooperative  infrastructure-side  vehicle-side
  * 'infrastructure-side' and 'vehicle-side' has 'image', 'velodyne', 'calib', and 'label', and data_info.json as follows. 
  * 'vehicle-side' label is in **Vehicle LiDAR Coordinate System**, while 'infrastructure-side' label is in **Infrastructure Virtual LiDAR Coordinate System**

    ├── infrastructure-side             # DAIR-V2X-C-I
        ├── image		    
            ├── {id}.jpg
        ├── velodyne                
            ├── {id}.pcd           
        ├── calib                 
            ├── camera_intrinsic            
                ├── {id}.json     
            ├── virtuallidar_to_world   
                ├── {id}.json      
            ├── virtuallidar_to_camera  
                ├── {id}.json      
        ├── label	
            ├── camera                  # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in image based on image frame time
                ├── {id}.json
            ├── virtuallidar            # Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects in point cloud based on point cloud frame time
                ├── {id}.json
        ├── data_info.json              # Relevant index information of Infrastructure data

 * The 'cooperative' folder contains the following files
    ├── cooperative                     # Coopetative Files
        ├── label_world                 # Vehicle-Infrastructure Cooperative (VIC) Annotation files
            ├── {id}.json           
        ├── data_info.json              # Relevant index information combined the Infrastructure data and the Vehicle data

There are four data folders under root '/data/cmpe249-fa22/DAIR-C':
 * 'cooperative-vehicle-infrastructure-vehicle-side-image' folder contains all images (6digit_id.jpg) in vehicle side.
 * 'cooperative-vehicle-infrastructure-vehicle-side-velodyne' folder contains all lidar files (6digit_id.pcd) in vehicle side.
 * 'cooperative-vehicle-infrastructure-infrastructure-side-image' folder contains all images (6digit_id.jpg) in infrastructure side.
 * 'cooperative-vehicle-infrastructure-infrastructure-side-velodyne' folder contains all lidar files (6digit_id.pcd) in infrastructure side.
 
 
Copy the split data (json files in 'https://github.com/AIR-THU/DAIR-V2X/tree/main/data/split_datas') to the data folder ('/data/cmpe249-fa22/DAIR-C')

Convert the dataset to KITTI format 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 'mydetector3d/datasets/dairv2x/dair2kitti.py', convert the vehicle-side data to Kitti format, set: 
 * 'source-root=/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/vehicle-side/'
 * 'target-root=/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti'
 * 'sourcelidarfolder=/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-vehicle-side-velodyne'
 * 'split-path=/data/cmpe249-fa22/DAIR-C/split_datas/single-vehicle-split-data.json'
 * 'sensor_view=vehicle'

The conversion process involve the following major steps:
 * First create kitti folder, then call **rawdata_copy** to copy images from source to target (kitti folder).
 * 'mykitti_pcd2bin': created new folder '/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/velodyne', convert pcd files in 'cooperative-vehicle-infrastructure-vehicle-side-velodyne' to bin files in Kitti 'velodyne' folder. Get xyz and intensity from pcd file, divide intensity/255, save xyz and new intensity to kitti velodyne bin file.
 * 'gen_lidar2cam', data_info=read_json(source_root/data_info.json), for each data in data_info, 
    * read 'calib/lidar_to_camera/id.json' and get Tr_velo_to_cam (3,4) 
    * read labels_path 'label/lidar/id.json', for each label in labels, 
       * get 'h, w, l, x, y, z, yaw_lidar', perform 'z = z - h / 2' get bottom_center
       * convert bottom_center to camera coordinate, get 'alpha, yaw' from **get_camera_3d_8points** 
       * use **convert_point** to get 'cam_x, cam_y, cam_z', and **set_label**
    * Write labels to 'tmp_file/label/lidar/id.json', get 'path_camera_intrinsic' and 'path_lidar_to_camera' under calib folder, call **gen_calib2kitti** get kitti calibration
 * use **json2kitti** to convert json label to kitti_label_root (/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/label_2/000000.txt)
    * change code in write_kitti_in_txt, save txt to '/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/label_2'
 * Generate calibration files, 
 * The converted kitti folder is '/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti'. The 'testing folder is empty', the image folder is not available in training, need to copy the images to training folder:
 
 .. code-block:: console
 
  (mycondapy39) [010796032@coe-hpc2 training]$ ls
  calib  label_2  velodyne
  (mycondapy39) [010796032@coe-hpc2 training]$ mkdir image_2
  (mycondapy39) [010796032@coe-hpc2 training]$ cd image_2/
  (mycondapy39) [010796032@coe-hpc2 image_2]$ cp /data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-vehicle-side-image/* .

In 'mydetector3d/datasets/dairv2x/dair2kitti.py', convert the infrastructure-side data to Kitti format, set: 
 * 'source-root=/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/infrastructure-side/'
 * 'target-root=/data/cmpe249-fa22/DAIR-C/infrastructure-side-point-cloud-kitti'
 * 'sourcelidarfolder=/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-infrastructure-side-velodyne'
 * 'split-path=/data/cmpe249-fa22/DAIR-C/split_datas/single-infrastructure-split-data.json'
 * 'sensor_view=infrastructure'

Created kitti folder "/data/cmpe249-fa22/DAIR-C/infrastructure-side-point-cloud-kitti"

.. code-block:: console

 (mycondapy39) [010796032@coe-hpc2 DAIR-C]$ cd infrastructure-side-point-cloud-kitti/
 (mycondapy39) [010796032@coe-hpc2 infrastructure-side-point-cloud-kitti]$ ls
 ImageSets  testing  training
 (mycondapy39) [010796032@coe-hpc2 infrastructure-side-point-cloud-kitti]$ cd training/
 (mycondapy39) [010796032@coe-hpc2 training]$ ls
 calib  label_2  velodyne
 (mycondapy39) [010796032@coe-hpc2 training]$ mkdir image_2 && cd image_2
 (mycondapy39) [010796032@coe-hpc2 image_2]$ cp /data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure-infrastructure-side-image/* .

Prepare the dataset 
~~~~~~~~~~~~~~~~~~~

Run dairkitti_dataset.py to generate the split files, infos, and gt_database.
 * run **create_split** option in dairkitti_dataset.py to create the split files (trainval.txt, train.txt, and val.txt) in 'ImageSets'
 * run **create_infos** to generate 'kitti_infos_xx.pkl' and call **create_groundtruth_database** to generate the gt_database
 
.. code-block:: console
 
  $ dairkitti_dataset.py
  gt_database sample: 12228/12228
  Database Car: 106628
  Database Motorcyclist: 14916
  Database Cyclist: 8845
  Database Trafficcone: 85790
  Database Pedestrian: 9060
  Database Tricyclist: 3286
  ---------------Data preparation Done---------------
  $ ls /data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/
  gt_database  kitti_dbinfos_train.pkl  kitti_infos_train.pkl     kitti_infos_val.pkl  training
  ImageSets    kitti_infos_test.pkl     kitti_infos_trainval.pkl  testing

In the **__getitem__** of dairkitti_dataset.py, gt_boxes_lidar is from 'location', 'dimensions', and 'rotation_y'

.. code-block:: console

  loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
  gt_names = annos['name']
  #create label [n,7] in camera coordinate boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
  gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
  gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

If this frame has no object, set gt_boxes_lidar empty:

.. code-block:: console

  if len(gt_names)==0:
       gt_boxes_lidar = np.zeros((0, 7))


OpenCOOD
------------------

Use `OpenCOOD <https://github.com/DerrickXuNu/OpenCOOD>`_ and ref `installation <https://opencood.readthedocs.io/en/latest/md_files/installation.html>`_ to setup the V2V cooperative 3D object detection framework (based on OpenPCDet) in Newalienware machine (with RTX3090)

.. code-block:: console

  (mycondapy39) lkk68@NEWALIENWARE C:\Users\lkk68\Documents\Developer>git clone https://github.com/DerrickXuNu/OpenCOOD.git
  (mycondapy39) lkk68@NEWALIENWARE C:\Users\lkk68\Documents\Developer\OpenCOOD>python setup.py develop
  #error: scipy 1.5.4 is installed but scipy>=1.8 is required by {'scikit-image'}
  $ pip install scipy -U
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
  opencood 0.1.0 requires matplotlib~=3.3.3, but you have matplotlib 3.7.1 which is incompatible.
  opencood 0.1.0 requires opencv-python~=4.5.1.48, but you have opencv-python 4.7.0.72 which is incompatible.
  opencood 0.1.0 requires scipy~=1.5.4, but you have scipy 1.10.1 which is incompatible.
  Successfully installed scipy-1.10.1

opv2v dataset is downloaded in '/data/cmpe249-fa22/OpenCOOD/opv2v_data_dumping', but there are errors in the dataset: "unzip:  cannot find zipfile directory in one of train.zip"
  
