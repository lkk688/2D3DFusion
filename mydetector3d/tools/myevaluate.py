from scipy.fft import fft, ifft, fftfreq, fftshift #solve the ImportError: /cm/local/apps/gcc/11.2.0/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

# from tensorboardX import SummaryWriter

from mydetector3d.tools.eval_utils import statistics_info #eval_one_epoch
from mydetector3d.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
# from mydetector3d.datasets import build_dataloader
# from mydetector3d.models import build_network
from mydetector3d.utils import common_utils

import pickle
import tqdm
from mydetector3d.models import load_data_to_gpu


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1" #"0,1"

#'/home/010796032/3DObject/modelzoo_openpcdet/pointpillar_7728.pth'
#'mydetector3d/tools/cfgs/kitti_models/pointpillar.yaml'
#'/home/010796032/3DObject/3DDepth/output/kitti_models/pointpillar/0413/ckpt/latest_model.pth'

#Second
#'mydetector3d/tools/cfgs/kitti_models/second_multihead.yaml'
#/home/010796032/3DObject/3DDepth/output/kitti_models/second_multihead/0415/ckpt/checkpoint_epoch_256.pth

#cfg_file: 'mydetector3d/tools/cfgs/kitti_models/second.yaml'
#--ckpt: '/home/010796032/3DObject/modelzoo_openpcdet/second_7862.pth'
#pretrained_model: None

#'mydetector3d/tools/cfgs/waymokitti_models/voxelnext_3class.yaml'
#'/data/cmpe249-fa22/Mymodels/waymokitti_models/voxelnext/0425/ckpt/checkpoint_epoch_128.pth'

#'mydetector3d/tools/cfgs/waymo_models/pointpillar_1x.yaml'
#'/data/cmpe249-fa22/Mymodels/waymo_models/pointpillar_1x/0427/ckpt/checkpoint_epoch_64.pth'

#'mydetector3d/tools/cfgs/waymo_models/myvoxelnext_ioubranch.yaml'
#'/data/cmpe249-fa22/Mymodels/waymo_models/myvoxelnext_ioubranch/0429/ckpt/checkpoint_epoch_256.pth'
#'mydetector3d/tools/cfgs/waymo_models/mysecond.yaml'

#'mydetector3d/tools/cfgs/waymokitti_models/second.yaml'
#'/data/cmpe249-fa22/Mymodels/waymokitti_models/second/0501/ckpt/checkpoint_epoch_64.pth'
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='mydetector3d/tools/cfgs/waymokitti_models/second.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='0426', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='/data/cmpe249-fa22/Mymodels/waymokitti_models/second/0501/ckpt/checkpoint_epoch_64.pth', help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    #parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    #parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    #parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', default=True, help='')
    parser.add_argument('--savebatchidx', type=int, default=1, help='Save one batch data to pkl for visualization')
    parser.add_argument('--infer_time', default=True, help='calculate inference latency') #action='store_true' true if specified

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem #Returns the substring from the beginning of filename: pointpillar
    #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[-2:-1])# get kitti_models

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

from mydetector3d.models.detectors.pointpillar import PointPillar
from mydetector3d.models.detectors.second_net import SECONDNet
from mydetector3d.models.detectors.voxelnext import VoxelNeXt
from mydetector3d.models.detectors.my3dmodel import My3Dmodel
__modelall__ = {
    #'Detector3DTemplate': Detector3DTemplate,
     'SECONDNet': SECONDNet,
    # 'PartA2Net': PartA2Net,
    # 'PVRCNN': PVRCNN,
     'PointPillar': PointPillar,
     'My3Dmodel': My3Dmodel,
     'VoxelNeXt': VoxelNeXt
}

from mydetector3d.datasets.kitti.kitti_dataset import KittiDataset
from mydetector3d.datasets.kitti.waymokitti_dataset import WaymoKittiDataset
from mydetector3d.datasets.waymo.waymo_dataset import WaymoDataset
from functools import partial
from torch.utils.data import DataLoader
__datasetall__ = {
    'KittiDataset': KittiDataset,
    'WaymoKittiDataset': WaymoKittiDataset,
    'WaymoDataset': WaymoDataset
}

def rundetection(dataloader, model, device, args, eval_output_dir, logger):
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    with torch.no_grad():
        dataset = dataloader.dataset
        class_names = dataset.class_names
        det_annos = []

        if getattr(args, 'infer_time', False):
            start_iter = int(len(dataloader) * 0.1)
            infer_time_meter = common_utils.AverageMeter()
        
        logger.info('*************** Start EVALUATION *****************')
        start_time = time.time()

        # #batch_dict data:
        # 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
        # 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
        # Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
        # Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
        # Voxel_num_points: (89196,)
        for i, batch_dict in enumerate(dataloader):
            #load_data_to_gpu(batch_dict)
            batch_dict.to(device)

            if getattr(args, 'infer_time', False):
                start_time = time.time()

            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict) #batch size array of record_dict{'pred_boxes'[N,7],'pred_scores'[N],'pred_labels'[N]}
                #ret_dict return: 'gt': 69, 'roi_0.3': 0, 'rcnn_0.3': 68, 'roi_0.5': 0, 'rcnn_0.5': 66, 'roi_0.7': 0, 'rcnn_0.7': 61
            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            statistics_info(cfg, ret_dict, metric, disp_dict) #disp_dict: 'infer_time': '87057.09(87057.09)', 'recall_0.3': '(0, 68) / 69'
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=eval_output_dir if args.save_to_file else None
            )#batch_size array, each dict is the Kitti annotation-like format dict (2D box is converted from 3D pred box)
            det_annos += annos #annos array: batchsize(16) pred_dict in each batch; det_annos array: all objects in all frames in the dataset
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

            if args.savebatchidx is not None and i==args.savebatchidx:
                #save the current batch data for later evaluation
                save_dict = {}
                save_dict['idx']=i
                save_dict['modelname']='second'
                save_dict['ckpt']=args.ckpt
                save_dict['cfg_file']=args.cfg_file
                save_dict['datasetname']='waymokitti'
                save_dict['batch_dict']=batch_dict
                save_dict['pred_dicts']=pred_dicts
                save_dict['annos']=annos
                resultfile='output/waymokitti_second_epoch64_onebatch_%s.pkl' % str(i)
                with open(resultfile, 'wb') as f:
                    pickle.dump(save_dict, f)
        progress_bar.close()

        ret_dict = {}
        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos: #each frame's results
            total_pred_objects += anno['name'].__len__()
        logger.info('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    return det_annos, ret_dict

def runevaluation(cfg, dataset, det_annos, class_names, final_output_dir, logger):
    
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC, #kitti
        output_path=final_output_dir
    )

    logger.info(result_str)
    #ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % final_output_dir)
    logger.info('****************Evaluation done.*****************')
    return result_str, result_dict


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dist_test = False
    total_gpus = 1
    device=torch.device('cuda:2')

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    #Folder format: output/kitti_models(EXP_GROUP_PATH)/pointpillar(TAG)/extra_tag
    #output_dir = cfg.ROOT_DIR / 'output' /  cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'txtresults' #'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    # logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    dataset_cfg = cfg.DATA_CONFIG
    class_names=cfg.CLASS_NAMES
    dataset = __datasetall__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=None,
        training=False,
        logger=logger,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
        shuffle=None, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=None)
    )

    #model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model_cfg=cfg.MODEL
    model_name=model_cfg.NAME
    num_class=len(cfg.CLASS_NAMES)
    model = __modelall__[model_name](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, 
                                pre_trained_path=args.pretrained_model)
    #model.cuda()
    model.to(device)
    model.eval()
    det_annos, ret_dict = rundetection(dataloader, model, device, args, eval_output_dir, logger)
    resultfile=output_dir / 'result.pkl'
    with open(resultfile, 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict =runevaluation(cfg, dataset, det_annos, class_names, eval_output_dir, logger)
    print(len(result_dict))

def evaluationonly(resultpath):
    args, cfg = parse_config()

    pklfile = open(os.path.join(resultpath, 'result.pkl'), 'rb')
    det_annos = pickle.load(pklfile)
    # close the file
    pklfile.close()

    dataset_cfg = cfg.DATA_CONFIG
    class_names=cfg.CLASS_NAMES
    dataset = __datasetall__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=None,
        training=False,
        logger=None,
    )

    # output_dir = Path(resultpath)
    # result_str, result_dict =runevaluation(cfg, dataset, det_annos, class_names, output_dir, None)
    # print(len(result_dict))

    output_dir = Path(resultpath)
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC, #kitti
        output_path=output_dir
    )
    print(result_str)
    print(result_dict)


    




if __name__ == '__main__':
    main()

    #evaluationonly('output/waymokitti_models/voxelnext/0425/')