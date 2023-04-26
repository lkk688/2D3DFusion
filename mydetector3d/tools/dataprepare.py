import yaml
from pathlib import Path
from easydict import EasyDict
from mydetector3d.datasets.kitti.kitti_dataset import create_kitti_infos

dataset_cfg = EasyDict(yaml.safe_load(open("mydetector3d/tools/cfgs/dataset_configs/kitti_dataset.yaml")))
create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=Path('/data/cmpe249-fa22/kitti'),
        save_path=Path('/data/cmpe249-fa22/kitti')
    )