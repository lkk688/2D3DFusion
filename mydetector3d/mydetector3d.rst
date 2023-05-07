mydetector3d training and evaluation
=====

.. _setup:

Trained Models
----------------------------

These three models are trained based on ** Waymo ** dataset in HPC2, the model saved path is '/data/cmpe249-fa22/Mymodels/waymo_models/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext/0427b/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext_ioubranch.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext_ioubranch/0429/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/mysecond.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/mysecond/0429/ckpt/checkpoint_epoch_128.pth'
    * Evaluation result saved in '/data/cmpe249-fa22/Mymodels/eval/waymo_models_mysecond_epoch128'

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

**annotations** dict contains: ['name'], ['truncated'], ['occluded'], ['alpha'], ['bbox'], ['dimensions']: lhw(camera) format, ['location'], ['rotation_y'], ['score'], ['difficulty'], among them

.. code-block:: console

 loc_lidar = calib.rect_to_lidar(loc)
 l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
 loc_lidar[:, 2] += h[:, 0] / 2
 gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
 annotations['gt_boxes_lidar'] = gt_boxes_lidar


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
~~~~~~~~~~~
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

Initialize the dataset during training
~~~~~~~~~~~
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
