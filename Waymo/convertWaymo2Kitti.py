from os import path as osp
import os
import Waymo2Kitti

folders = ['training_0000', 'training_0001']
root_path="/DataDisk1/WaymoDataset"
out_dir="/DataDisk1/WaymoKitti"
workers=4
for i, split in enumerate(folders):
    #load_dir = osp.join(root_path, 'waymo_format', split)
    load_dir = osp.join(root_path, split)
    if split == 'validation':
        save_dir = osp.join(out_dir, 'validation')
    else:
        save_dir = osp.join(out_dir, 'training', split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    converter = Waymo2Kitti.Waymo2KITTI(
        load_dir,
        save_dir,
        workers=workers,
        test_mode=(split == 'test'))
    converter.convert()