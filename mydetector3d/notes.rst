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
  #. mycreateImageSet: Create the folder 'ImageSets' for the list of train val split file names under '/data/cmpe249-fa22/Waymo132/ImageSets/'
  
  #. mygeninfo: create info files based on the provided folder list, the processed_data_tag='train0to9'
  
    * call waymo_utils.process_single_sequence for each tfrecord sequence file, all returned infos dict list are saved in train0to9_infos_train.pkl under root folder '/data/cmpe249-fa22/Waymo132/'
    * waymo_utils.process_single_sequence created one folder for each sequence under the folder '/data/cmpe249-fa22/Waymo132/train0to9'. One pkl file contains list of all sequence info is saved, including annotations (via generate_labels). 
    * generate_labels in mydetector3d/datasets/waymo/waymo_utils.py utilize waymo frame.laser_labels for box annatation, loc = [box.center_x, box.center_y, box.center_z], dimensions.append([box.length, box.width, box.height])
    * save_lidar_points save each frame's lidar data as one npy file (frame index as the name) under the sequence folder, 3d points in vehicle frame.
    
  #. mygengtdb: create the groundtruth database via create_waymo_gt_database function
  
    * call dataset.create_groundtruth_database for 'train' split
    * created '%s_gt_database_%s_sampled_%d_global.npy' and '%s_waymo_dbinfos_%s_sampled_%d.pkl' under the root folder
    * created '%s_gt_database_%s_sampled_%d' folder under the root




.. code-block:: console

  

Use Slurm to request one GPU node, and setup required paths
