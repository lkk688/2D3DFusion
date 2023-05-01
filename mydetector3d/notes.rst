mydetector3d training and evaluation
=====

.. _setup:

Waymo Dataset Trained Models
------------

These three models are trained based on Waymo dataset in HPC2, the model saved path is '/data/cmpe249-fa22/Mymodels/waymo_models/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext/0427b/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/myvoxelnext_ioubranch.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext_ioubranch/0429/ckpt/'
  * cfg_file='mydetector3d/tools/cfgs/waymo_models/mysecond.yaml', ckpt file in '/data/cmpe249-fa22/Mymodels/waymo_models/mysecond/0429/ckpt/'

Waymo Dataset Process
------------

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
    data_dict = self.point_feature_encoder.forward(data_dict) #do feature encoder for points
    data_dict = self.data_processor.forward #pre-processing for the points remove out of range ponts, shuffle, and convert to voxel
  

  

Use Slurm to request one GPU node, and setup required paths
