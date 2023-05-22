
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
    cam_front_data