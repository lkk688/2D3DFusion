# ref: https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/tool/exporter.py

import glob
import onnx
import torch
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify #require pip install onnxruntime
from mydetector3d.utils import common_utils
from mydetector3d.models import build_network, load_data_to_gpu
from mydetector3d.datasets import DatasetTemplate
from mydetector3d.config import cfg, cfg_from_yaml_file
from simplifier_onnx import simplify_preprocess, simplify_postprocess

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3" #"0,1"

#from exporter_paramters import export_paramters as export_paramters
def export_paramters(cfg):
    CLASS_NAMES = []
    CLASS_NUM = 0
    rangMinX = 0
    rangMinY = 0
    rangMinZ = 0
    rangMaxX = 0
    rangMaxY = 0
    rangMaxZ = 0
    VOXEL_SIZE = []
    MAX_POINTS_PER_VOXEL = 0
    MAX_NUMBER_OF_VOXELS = 0
    NUM_POINT_FEATURES = 0
    NUM_BEV_FEATURES = 0
    DIR_OFFSET = 0
    DIR_LIMIT_OFFSET = 0
    NUM_DIR_BINS = 0
    anchor_sizes = []
    anchor_bottom_heights = []
    SCORE_THRESH = 0
    NMS_THRESH = 0

    CLASS_NAMES = cfg.CLASS_NAMES
    CLASS_NUM = len(CLASS_NAMES)

    rangMinX = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0]
    rangMinY = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[1]
    rangMinZ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[2]
    rangMaxX = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3]
    rangMaxY = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[4]
    rangMaxZ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE[5]

    for item in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if (item.NAME == "transform_points_to_voxels"):
            VOXEL_SIZE = item.VOXEL_SIZE
            MAX_POINTS_PER_VOXEL = item.MAX_POINTS_PER_VOXEL
            MAX_NUMBER_OF_VOXELS = item.MAX_NUMBER_OF_VOXELS.test

    for item in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST:
        if (item.NAME == "gt_sampling"):
            NUM_POINT_FEATURES = item.NUM_POINT_FEATURES

    NUM_BEV_FEATURES = cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
    DIR_OFFSET = cfg.MODEL.DENSE_HEAD.DIR_OFFSET
    DIR_LIMIT_OFFSET = cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET
    NUM_DIR_BINS = cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS

    for item in cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG:
        for anchor in np.array(item.anchor_sizes).flatten():
            anchor_sizes.append(float(anchor))
        anchor_sizes.append(float(item.anchor_rotations[0]))
        for anchor in np.array(item.anchor_sizes).flatten():
            anchor_sizes.append(float(anchor))
        anchor_sizes.append(float(item.anchor_rotations[1]))
        for anchor_height in np.array(item.anchor_bottom_heights).flatten():
            anchor_bottom_heights.append(anchor_height)

    SCORE_THRESH = cfg.MODEL.POST_PROCESSING.SCORE_THRESH
    NMS_THRESH = cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH

    # dump paramters to params.h
    fo = open("params.h", "w")
    #fo.write(License+"\n")
    fo.write("#ifndef PARAMS_H_\n#define PARAMS_H_\n")
    fo.write("const int MAX_VOXELS = "+str(MAX_NUMBER_OF_VOXELS)+";\n")

    fo.write("class Params\n{\n  public:\n")

    fo.write("    static const int num_classes = "+str(CLASS_NUM)+";\n")
    class_names_list = "    const char *class_name [num_classes] = { "
    for CLASS_NAME in CLASS_NAMES:
        class_names_list = class_names_list + "\""+CLASS_NAME+"\","
    class_names_list = class_names_list + "};\n"
    fo.write(class_names_list)

    fo.write("    const float min_x_range = "+str(float(rangMinX))+";\n")
    fo.write("    const float max_x_range = "+str(float(rangMaxX))+";\n")
    fo.write("    const float min_y_range = "+str(float(rangMinY))+";\n")
    fo.write("    const float max_y_range = "+str(float(rangMaxY))+";\n")
    fo.write("    const float min_z_range = "+str(float(rangMinZ))+";\n")
    fo.write("    const float max_z_range = "+str(float(rangMaxZ))+";\n")

    fo.write("    // the size of a pillar\n")
    fo.write("    const float pillar_x_size = " +
             str(float(VOXEL_SIZE[0]))+";\n")
    fo.write("    const float pillar_y_size = " +
             str(float(VOXEL_SIZE[1]))+";\n")
    fo.write("    const float pillar_z_size = " +
             str(float(VOXEL_SIZE[2]))+";\n")

    fo.write("    const int max_num_points_per_pillar = " +
             str(MAX_POINTS_PER_VOXEL)+";\n")

    fo.write("    const int num_point_values = "+str(NUM_POINT_FEATURES)+";\n")
    fo.write("    // the number of feature maps for pillar scatter\n")
    fo.write("    const int num_feature_scatter = " +
             str(NUM_BEV_FEATURES)+";\n")

    fo.write("    const float dir_offset = "+str(float(DIR_OFFSET))+";\n")
    fo.write("    const float dir_limit_offset = " +
             str(float(DIR_LIMIT_OFFSET))+";\n")

    fo.write("    // the num of direction classes(bins)\n")
    fo.write("    const int num_dir_bins = "+str(NUM_DIR_BINS)+";\n")

    fo.write("    // anchors decode by (x, y, z, dir)\n")
    fo.write("    static const int num_anchors = num_classes * 2;\n")
    fo.write("    static const int len_per_anchor = 4;\n")

    anchor_str = "    const float anchors[num_anchors * len_per_anchor] = {\n"
    anchor_str += "      "
    count = 0
    for item in anchor_sizes:
        anchor_str = anchor_str + str(float(item)) + ","
        count += 1
        if ((count % 4) == 0):
            anchor_str += "\n      "
    anchor_str = anchor_str + "};\n"
    fo.write(anchor_str)

    anchor_heights = "    const float anchor_bottom_heights[num_classes] = {"
    for item in anchor_bottom_heights:
        anchor_heights = anchor_heights + str(float(item)) + ","
    anchor_heights = anchor_heights + "};\n"
    fo.write(anchor_heights)
    fo.write("    // the score threshold for classification\n")
    fo.write("    const float score_thresh = "+str(float(SCORE_THRESH))+";\n")
    fo.write("    const float nms_thresh = "+str(float(NMS_THRESH))+";\n")

    fo.write(
        '''    const int max_num_pillars = MAX_VOXELS;
    const int pillarPoints_bev = max_num_points_per_pillar * max_num_pillars;
    // the detected boxes result decode by (x, y, z, w, l, h, yaw)
    const int num_box_values = 7;
    // the input size of the 2D backbone network
    const int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
    const int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
    const int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
    // the output size of the 2D backbone network
    const int feature_x_size = grid_x_size / 2;
    const int feature_y_size = grid_y_size / 2;\n'''
    )

    fo.write("    Params() {};\n};\n")
    fo.write("#endif\n")
    fo.close()


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(
            str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='mydetector3d/tools/cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/data/cmpe249-fa22/kitti/testing/velodyne',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/010796032/3DObject/modelzoo_openpcdet/pointpillar_7728.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    export_paramters(cfg)
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    np.set_printoptions(threshold=np.inf)
    with torch.no_grad():

        MAX_VOXELS = 10000

        dummy_voxels = torch.zeros(
            (MAX_VOXELS, 32, 4),
            dtype=torch.float32,
            device='cuda:0') #[10000, 32, 4]

        dummy_voxel_idxs = torch.zeros(
            (MAX_VOXELS, 4),
            dtype=torch.int32,
            device='cuda:0') #[10000, 4]

        dummy_voxel_num = torch.zeros(
            (1),
            dtype=torch.int32,
            device='cuda:0')

        dummy_input = dict()
        dummy_input['voxels'] = dummy_voxels #[10000, 32, 4]
        dummy_input['voxel_num_points'] = dummy_voxel_num #number
        dummy_input['voxel_coords'] = dummy_voxel_idxs #[10000, 4]
        dummy_input['batch_size'] = 1

        my_input = dict()
        my_input['batch_dict'] = dummy_input

        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print(pred_dicts[0])
            pred_dicts, _ = model(data_dict)
            break
    
        #Error: forward() takes 2 positional arguments but 4 were given
        # torch.onnx.export(model,                   # model being run
        #   (dummy_voxels, dummy_voxel_num, dummy_voxel_idxs), # model input (or a tuple for multiple inputs)
        #   "./pointpillar.onnx",    # where to save the model (can be a file or file-like object)
        #   export_params=True,        # store the trained parameter weights inside the model file
        #   opset_version=11,          # the ONNX version to export the model to
        #   do_constant_folding=True,  # whether to execute constant folding for optimization
        #   keep_initializers_as_inputs=True,
        #   input_names = ['input', 'voxel_num_points', 'coords'],   # the model's input names
        #   output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'], # the model's output names
        #   )

        torch.onnx.export(model,       # model being run
                          # model input (or a tuple for multiple inputs) 
                          my_input, 
                          # where to save the model (can be a file or file-like object)
                          "./pointpillar_raw.onnx",
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          keep_initializers_as_inputs=True,
                          # the model's input names
                          input_names=['voxels', 'voxel_num', 'voxel_idxs'],
                          # the model's output names
                          output_names=['cls_preds',
                                        'box_preds', 'dir_cls_preds'],
                        )

        onnx_raw = onnx.load("./pointpillar_raw.onnx")  # load onnx model
        onnx_trim_post = simplify_postprocess(onnx_raw)

        onnx_simp, check = simplify(onnx_trim_post)
        assert check, "Simplified ONNX model could not be validated"

        onnx_final = simplify_preprocess(onnx_simp)
        onnx.save(onnx_final, "pointpillar.onnx")
        print('finished exporting onnx')

    logger.info('[PASS] ONNX EXPORTED.')


if __name__ == '__main__':
    main()
