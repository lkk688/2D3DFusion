
from nuscenes.nuscenes import NuScenes

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='mydetector3d/tools/cfgs/dataset_configs/nuscenes_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--datapath', type=str, default='/data/cmpe249-fa22/nuScenes/nuScenesv1.0-mini', help='specify the path of dataset')
    parser.add_argument('--func', type=str, default='create_groundtruth', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', default=True, help='use camera or not')
    args = parser.parse_args()
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version
    
    nusc = NuScenes(version='v1.0-mini', dataroot=args.datapath, verbose=True)
    nusc.list_scenes() #1000 scenes, 20s each

    #get one scene
    my_scene = nusc.scene[0]
    #sample (annotated keyframe) is annotated 2Hz
    first_sample_token = my_scene['first_sample_token'] #'ca9a282c9e77460f8360f564131a8af5'
    #rendering
    nusc.render_sample(first_sample_token)

    #get sample's metadata
    my_sample = nusc.get('sample', first_sample_token) #dict

    #list all related sample_data keyframes and sample_annotation associated with a sample
    nusc.list_sample(my_sample['token'])
    #use data key to access full sensor data
    print(my_sample['data'])
    #check the metadata of a sample_data from CAM_FRONT
    sensor = 'CAM_FRONT'
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    print(cam_front_data)
    nusc.render_sample_data(cam_front_data['token'])#get 3D box on front image

    #sample_annotation is the boundin gbox defining the object position in global coordinate system
    my_annotation_token = my_sample['anns'][18]
    my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
    print(my_annotation_metadata)
    nusc.render_annotation(my_annotation_token) #camera and lidar bbox annotation

    #check instance (i.e., the object)
    my_instance = nusc.instance[599]
    print(my_instance)
    instance_token = my_instance['token']
    nusc.render_instance(instance_token)
    print("First annotated sample of this instance:")
    nusc.render_annotation(my_instance['first_annotation_token'])
    print("Last annotated sample of this instance")
    nusc.render_annotation(my_instance['last_annotation_token'])

    #category
    nusc.list_categories()
    print(nusc.category[9])

    #sensors: 1 Lidar, 5 Radar, 6 camera
    print(nusc.sensor)
    print(nusc.sample_data[10]) #'channel' shows the sensor
    #`calibrated_sensor` consists of the definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle
    print(nusc.calibrated_sensor[0]) #get translation and the rotation with respect to the ego vehicle body frame
    #ego_pose` contains information about the location (encoded in `translation`) and the orientation (encoded in `rotation`) of the ego vehicle, with respect to the global coordinate system.
    print(nusc.ego_pose[0]) #ego_pose to sample_data record are one-to-one 

    #map is binary semantic masks
    print("There are {} maps masks in the loaded dataset".format(len(nusc.map)))
    nusc.map[0]

    my_sample = nusc.sample[10]
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=True)
    #Radar data
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')

    my_sample = nusc.sample[20]
    nusc.render_sample(my_sample['token'])

    #get a particular sensor
    nusc.render_sample_data(my_sample['data']['CAM_FRONT'])

    #aggregate multiple sweeps of point cloud
    nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
    nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)

    from nuscenes.utils.data_classes import RadarPointCloud
    RadarPointCloud.disable_filters()
    nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
    RadarPointCloud.default_filters()

    nusc.render_annotation(my_sample['anns'][22])

    my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
    #render video for front camera
    nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
    #render video for all camera
    nusc.render_scene(my_scene_token)

    #visualize all scenes on the map for a particular location
    nusc.render_egoposes_on_map(log_location='singapore-onenorth')