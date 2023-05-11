mydetector3d training and evaluation
===================================

.. _setup:

Trained Models
----------------------------

These three models are trained based on ** Waymo ** dataset (Waymo132/train0to9) in HPC2, the model saved path is '/data/cmpe249-fa22/Mymodels/waymo_models/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext/0427b/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext_ioubranch.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext_ioubranch/0429/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/mysecond.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/mysecond/0429/ckpt/checkpoint_epoch_128.pth'
     * evaluation result saved in '/data/cmpe249-fa22/Mymodels/eval/waymo_models_mysecond_epoch128'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/my3dmodel.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/my3dmodel/0507/ckpt/checkpoint_epoch_128.pth'
     * evaluation result saved in '/data/cmpe249-fa22/Mymodels/eval/waymo_models_my3dmodel_epoch128'

Models are trained based on the complete Waymo dataset (Waymo132/trainall) in HPC2
* cfg_file='mydetector3d/tools/cfgs/waymo_models/my3dmodel.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/my3dmodel/0508/ckpt/checkpoint_epoch_256.pth' continue training from 129-256 based on ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/my3dmodel/0507/ckpt/checkpoint_epoch_128.pth'. 
    * Evaluation result is saved to /data/cmpe249-fa22/Mymodels/eval/waymo_models_my3dmodel_epoch256/txtresults.

.. code-block:: console

  Car AP@0.70, 0.70, 0.70:
   bbox AP:91.7851, 91.7851, 91.7851
   bev  AP:68.3034, 68.3034, 68.3034
   3d   AP:49.0174, 49.0174, 49.0174
   aos  AP:50.76, 50.76, 50.76
  Pedestrian AP@0.50, 0.50, 0.50:
   bbox AP:89.7635, 89.7635, 89.7635
   bev  AP:55.1775, 55.1775, 55.1775
   3d   AP:50.3953, 50.3953, 50.3953
   aos  AP:45.93, 45.93, 45.93
  Cyclist AP@0.50, 0.50, 0.50:
   bbox AP:64.8413, 64.8413, 64.8413
   bev  AP:51.8248, 51.8248, 51.8248
   3d   AP:48.8936, 48.8936, 48.8936
   aos  AP:51.74, 51.74, 51.74
 
 
* cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext/0509/ckpt/checkpoint_epoch_128.pth', trained from 0. 
    * Evaluation result is saved to /data/cmpe249-fa22/Mymodels/eval/waymo_models_myvoxelnext_epoch128/txtresults

.. code-block:: console

  Car AP@0.70, 0.70, 0.70:
  bbox AP:96.9390, 96.9390, 96.9390
  bev  AP:71.0638, 71.0638, 71.0638
  3d   AP:57.9034, 57.9034, 57.9034
  aos  AP:57.86, 57.86, 57.86
  Pedestrian AP@0.50, 0.50, 0.50:
  bbox AP:93.3127, 93.3127, 93.3127
  bev  AP:67.9591, 67.9591, 67.9591
  3d   AP:61.6305, 61.6305, 61.6305
  aos  AP:52.90, 52.90, 52.90
  Cyclist AP@0.50, 0.50, 0.50:
  bbox AP:80.0512, 80.0512, 80.0512
  bev  AP:70.7396, 70.7396, 70.7396
  3d   AP:69.7192, 69.7192, 69.7192
  aos  AP:67.86, 67.86, 67.86

New models are trained based on our converted ** WaymoKitti** dataset in HPC2, the model save path is '/data/cmpe249-fa22/Mymodels/waymokitti_models/'
  * cfg_file='mydetector3d/tools/cfgs/waymokitti_models/pointpillar.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymokitti_models/pointpillar/0504/ckpt/checkpoint_epoch_128.pth'
  * cfg_file='mydetector3d/tools/cfgs/waymokitti_models/second.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymokitti_models/second/0502/ckpt/checkpoint_epoch_128.pth'
  * cfg_file='mydetector3d/tools/cfgs/waymokitti_models/voxelnext_3class.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymokitti_models/voxelnext_3class/0430/ckpt/checkpoint_epoch_72.pth'
  * cfg_file='mydetector3d/tools/cfgs/waymokitti_models/my3dmodel.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymokitti_models/my3dmodel/0505/ckpt/latest_model.pth'

Model Evaluation
----------------
Evaluation results performed by 'myevaluatev2.py' are all saved in the folder of '/data/cmpe249-fa22/Mymodels/eval/', the naming of the folder is 'datasetname'+'modelname'+'epochnumber'

In each folder, e.g., '/data/cmpe249-fa22/Mymodels/eval/waymokitti_dataset_mysecond_epoch128/', there are three results (.pkl) saved by the detection function
  * result.pkl file: the detection results in kitti format, i.e., det_annos array. Used for evaluation
  * ret_dicts.pkl: the detection results, groundtruth, and inference time for each frame
  * txtresults: detection results saved in Kitti format, the 2D bounding box is converted from 3D bounding box.
  * waymokitti_dataset_mysecond_epoch128_frame_1.pkl (naming: 'datasetname_model_name_epochnumber_framenumber') saved whole frame data with original Lidar points and groundtruth. Input this pkl file to 'visonebatch.py' to visualize the 3D detection results

The following results are saved by the evaluation function **runevaluation**
  * result_str txt file: kitti evaluation results
  * result_dict.pkl: recall related evaluation data

In ** runevaluation ** , input "det_annos" from detection results
  * get infos from dataset.infos, each anno dict in det_annos contain the following keys: 'point_cloud', 'frame_id', 'metadata', 'image', 'annos', 'pose', and 'num_points_of_each_lidar' (5 Lidars)
  * The 'annos' key contain key 'gt_boxes_lidar' (63,9), 'dimensions', 'location', 'heading_angles' ,...
  * The 'annos' part are convert to **eval_gt_annos** via the following code

  .. code-block:: console

    eval_det_annos = copy.deepcopy(det_annos) # contains 'boxes_lidar' (N,7) key
    eval_gt_annos = [copy.deepcopy(info['annos']) for info in datainfo] # contains 'gt_boxes_lidar' (N,7) key
    transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
    transform_annotations_to_kitti_format(
                    eval_gt_annos, map_name_to_kitti=map_name_to_kitti, info_with_fakelidar = False)
    result_str, result_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

Kitti Dataset Process
-----------------------------
Run **create_kitti_infos** in 'mydetector3d/datasets/kitti/kitti_dataset.py', create 'kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_trainval.pkl', and 'kitti_infos_test.pkl' based on split file
 * call dataset.get_infos to generate each info.pkl file, process each file in sample_id_list via **process_single_scene**, save these infos

.. code-block:: console

 pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
 info['point_cloud'] = pc_info
 image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
 info['image'] = image_info
 info['calib'] = calib_info
 info['annos'] = annotations

Get all labels in obj_list via **self.get_label(sample_idx)**, where each obj is

.. code-block:: console

  return object3d_kitti.get_objects_from_label(label_file)
                          |-------[Object3d(line) for line in lines]
                                    |-----in mydetector3d/utils/object3d_kitti.py

**annotations** is created from the obj_list, and each dict contains: ['name'], ['truncated'], ['occluded'], ['alpha'], ['bbox'], ['dimensions']: lhw(camera) format, ['location'], ['rotation_y'], ['score'], ['difficulty']
  * 'name' is class name string from obj.cls_type
  * 'truncated' (0 non-truncated ~ 1 truncated), 'occluded' (0 fully visible,1,2,3 unknown), 'alpha' (observation angle -pi~pi) are float from original kitti label txt
    * alpha considers the vector from the camera center to the object center
    * alpha is zero when this object is located along the Z-axis (front) of the camera
  * 'bbox' is from obj.box2d label[4-7]: left, top, right, bottom image pixel coordinate (int)
  * 'dimensions' is 3d object size in meters [obj.l, obj.h, obj.w]
    * obj.l is from label[10] length
    * obj.h is from label[9] width
    * obj.w is from label[8] height
  * 'location' is from obj.loc (label[11-13]) xyz in camera coordinate
  * 'rotation_y' from label[14] Rotation ry around Y-axis (to the ground) in camera coordinates [-pi..pi]
  * 'difficulty' is calculated by **get_kitti_obj_level** based on the box2d height (pixel size>40 means Easy)

These **annotations**  values are further processed to convert the loc from camera rect coordinate to Lidar coordinate, and move the Z height of the loc_lidar (shift objects' center coordinate (original 0) from box bottom to the center)

.. code-block:: console

 loc_lidar = calib.rect_to_lidar(loc)
 l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
 loc_lidar[:, 2] += h[:, 0] / 2
 gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
 annotations['gt_boxes_lidar'] = gt_boxes_lidar

Where "-(np.pi / 2 + rots" is convert kitti camera rot angle definition (camera x-axis, clockwise is positive) to pcdet lidar rot angle definition (Lidar X-axis, clockwise is negative).

My Waymokitti Dataset Process
-----------------------------
My Waymokitti Dataset saved in '/data/cmpe249-fa22/WaymoKitti/4c_train5678'

.. code-block:: console

(mycondapy39) [010796032@coe-hpc2 cmpe249-fa22]$ ls /data/cmpe249-fa22/WaymoKitti/4c_train5678/
ImageSets   training                 waymo_gt_database      waymo_infos_trainval.pkl
ImageSets2  waymo_dbinfos_train.pkl  waymo_infos_train.pkl  waymo_infos_val.pkl

Converted Waymo dataset to Kitti format via 'Waymo2KittiAsync.py' in 'https://github.com/lkk688/WaymoObjectDetection', run the following code 

  .. code-block:: console
  
  [DatasetTools]$ python Waymo2KittiAsync.py
  [DatasetTools]$ python mycreatewaymoinfo.py --createsplitfile_only
  [DatasetTools]$ python mycreatewaymoinfo.py --createinfo_only
 
The groundtruth db generation is done in https://github.com/lkk688/mymmdetection3d

In **mycreatewaymoinfo.py**, createinfo_only will call **get_waymo_image_info** in 'https://github.com/lkk688/WaymoObjectDetection/blob/master/DatasetTools/myWaymoinfo_utils.py', it will create the following info
Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4 #6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }

Created a new dataset file 'mydetector3d/datasets/kitti/waymokitti_dataset.py' based on kitti_dataset.py.

Waymo Dataset Process
--------------------

Prepare the dataset 
~~~~~~~~~~~~~~~~~~~
In 'mydetector3d/datasets/waymo/waymo_dataset.py', specify the '--func' in main to select different preprocessing functions.
  * mycreateImageSet: Create the folder 'ImageSets' for the list of train val split file names under '/data/cmpe249-fa22/Waymo132/ImageSets/'
  * ** mygeninfo **: create info files based on the provided folder list, the processed_data_tag='train0to9'  
  * ** mygengtdb **: create the groundtruth database via create_waymo_gt_database function
  
In ** mygeninfo ** function:
    #. call waymo_utils.process_single_sequence for each tfrecord sequence file, all returned infos dict list are saved in train0to9_infos_train.pkl under root folder '/data/cmpe249-fa22/Waymo132/'
    #. waymo_utils.process_single_sequence created one folder for each sequence under the folder '/data/cmpe249-fa22/Waymo132/train0to9'. One pkl file contains list of all sequence info is saved, including annotations (via generate_labels). 
      * generate_labels in mydetector3d/datasets/waymo/waymo_utils.py utilize waymo frame.laser_labels for box annatation, loc = [box.center_x, box.center_y, box.center_z], dimensions.append([box.length, box.width, box.height])
      * save_lidar_points save each frame's lidar data as one npy file (frame index as the name) under the sequence folder, 3d points in vehicle frame.
    
In ** mygengtdb ** function->create_waymo_gt_database:
    #. call dataset.create_groundtruth_database (in waymo_dataset.py) for 'train' split
      * created '%s_gt_database_%s_sampled_%d_global.npy' (stacked_gt_points) and '%s_waymo_dbinfos_%s_sampled_%d.pkl' (array of dbinfo dict) under the root folder
      * each dbinfo is the following dict, each item is the groundtruth object with its gt_boxes and gt_points

      .. code-block:: console

       db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                                     'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                                     'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

      * created '%s_gt_database_%s_sampled_%d' folder under the root

Prepare all dataset
~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

 (mycondapy39) [010796032@cs001 waymo]$ python waymo_dataset.py --func 'mycreateImageSet'
 Total files: 648
 Train size: (518, 1)
 Val size: (130, 1)
 Done in /data/cmpe249-fa22/Waymo132/ImageSets/trainval.txt
 Done in /data/cmpe249-fa22/Waymo132/ImageSets/train.txt
 Done in /data/cmpe249-fa22/Waymo132/ImageSets/val.txt
 (mycondapy39) [010796032@cs001 waymo]$ python waymo_dataset.py --func 'mygeninfo'
 totoal number of files: 648
 (mycondapy39) [010796032@cs001 3DDepth]$ python mydetector3d/datasets/waymo/waymo_dataset.py --func 'mygengtdb'
  Total samples for Waymo dataset: 6485
  ---------------Start create groundtruth database for data augmentation---------------
  2023-05-08 18:06:49,870   INFO  Loading Waymo dataset
  2023-05-08 18:07:23,908   INFO  Total skipped info 0
  2023-05-08 18:07:23,908   INFO  Total samples for Waymo dataset: 25867
  Database Vehicle: 244715
  Database Pedestrian: 231457
  Database Cyclist: 11475                                                                                                
  ---------------Data preparation Done---------------

Initialize the dataset during training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialize class DatasetTemplate (in dataset.py), setup three processors specified in "DATA_PROCESSOR" section of the configuration file "mydetector3d/tools/cfgs/dataset_configs/mywaymo_dataset.yaml"
  * point_feature_encoder (based on dataset_cfg.POINT_FEATURE_ENCODING), 
  * data_augmentor (based on dataset_cfg.DATA_AUGMENTOR), 
  * data_processor (based on dataset_cfg.DATA_PROCESSOR). Get grid_size and voxel_size from data_processor.

  .. code-block:: console

  self.grid_size = self.data_processor.grid_size #[1504, 1504, 40] = POINT_CLOUD_RANGE/voxel_size
  self.voxel_size = self.data_processor.voxel_size #[0.1, 0.1, 0.15]meters

Initialize class WaymoDataset in 'mydetector3d/datasets/waymo/waymo_dataset.py', read infos[] via include_waymo_data function
  * In ** include_waymo_data ** function: Iterate through sample_sequence_list (all tfrecord files), load pkl file as infos in each sequence folder, add all together to infos[].

In **  __getitem__ ** function
  * Get point cloud info pc_info, then get the lidar points [N,5] [x, y, z, intensity, elongation]
  
  .. code-block:: console
   
   pc_info = info['point_cloud']
   sequence_name = pc_info['lidar_sequence']
   sample_idx = pc_info['sample_idx']
   points = self.get_lidar(sequence_name, sample_idx) #load the npy file, limit the intensity from -1 to 1
   input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

  * Get 'annos' in info
  
  .. code-block:: console
  
   gt_boxes_lidar = annos['gt_boxes_lidar'] #[N,9]
   gt_boxes_lidar = gt_boxes_lidar[:, 0:7] #[54,8] not use speed information
   #FILTER_EMPTY_BOXES_FOR_TRAIN
   input_dict.update({
                'gt_names': annos['name'], #class string names [54,]
                'gt_boxes': gt_boxes_lidar, #[54,7]
                'num_points_in_gt': annos.get('num_points_in_gt', None) #[54,]
            })

  * Call data_dict = self.prepare_data(data_dict=input_dict) (DatasetTemplate) 
  
   .. code-block:: console
   
    data_dict = self.data_augmentor.forward # perform data augmentation
    data_dict['gt_boxes'] = gt_boxes #Filter gt_boxes, convert gt_names to index and add to gt_boxes last column [Ngt,7]->[Ngt,8]
    data_dict = self.point_feature_encoder.forward(data_dict) #do feature encoder for points [N,5], only add use_lead_xyz=True
    data_dict = self.data_processor.forward #pre-processing for the points remove out of range ponts, shuffle, and convert to voxel (transform_points_to_voxels in data_processor.py)
  
  * transform_points_to_voxels in data_processor.py
  
   .. code-block:: console
  
    voxel_output = self.voxel_generator.generate(points) # get voxels (64657, 5, 5), coordinates (64657, 3), num_points (64657,)
    data_dict['voxels'] = voxels
    data_dict['voxel_coords'] = coordinates
    data_dict['voxel_num_points'] = num_points
  
 * get the final data_dict
  #. 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
  #. 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
  #. Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
  #. Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
  #. Voxel_num_points: (89196,)

Start the training for all waymo data

.. code-block:: console

  (mycondapy39) [010796032@cs001 3DDepth]$ python mydetector3d/tools/mytrain.py
  2023-05-08 19:16:49,940   INFO  cfg_file         mydetector3d/tools/cfgs/waymo_models/my3dmodel.yaml
  2023-05-08 19:16:49,940   INFO  batch_size       8
  2023-05-08 19:16:49,940   INFO  epochs           256
  2023-05-08 19:16:49,940   INFO  workers          4
  2023-05-08 19:16:49,940   INFO  extra_tag        0508
  2023-05-08 19:16:49,940   INFO  ckpt             /data/cmpe249-fa22/Mymodels/waymo_models/my3dmodel/0507/ckpt/checkpoint_epoch_128.pth
  2023-05-08 19:16:49,967   INFO  ----------- Create dataloader & network & optimizer -----------
  2023-05-08 19:16:53,197   INFO  Database filter by min points Vehicle: 244715 => 209266
  2023-05-08 19:16:53,222   INFO  Database filter by min points Pedestrian: 231457 => 196642
  2023-05-08 19:16:53,225   INFO  Database filter by min points Cyclist: 11475 => 10211
  2023-05-08 19:16:53,248   INFO  Database filter by difficulty Vehicle: 209266 => 209266
  2023-05-08 19:16:53,271   INFO  Database filter by difficulty Pedestrian: 196642 => 196642
  2023-05-08 19:16:53,272   INFO  Database filter by difficulty Cyclist: 10211 => 10211
  2023-05-08 19:16:53,323   INFO  Loading Waymo dataset
  2023-05-08 19:16:54,998   INFO  Total skipped info 0
  2023-05-08 19:16:54,998   INFO  Total samples for Waymo dataset: 25867
  2023-05-08 19:16:54,998   INFO  Total sampled samples for Waymo dataset: 5174
  Num point features initial 5
  Num point features after VFE 64
  num_bev_features features after BEV 64
  num_bev_features features after backbone2d 384

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
 * 'mykitti_pcd2bin': created new folder '/data/cmpe249-fa22/DAIR-C/single-vehicle-side-point-cloud-kitti/training/velodyne', convert pcd files in 'cooperative-vehicle-infrastructure-vehicle-side-velodyne' to bin files in Kitti 'velodyne' folder.
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
