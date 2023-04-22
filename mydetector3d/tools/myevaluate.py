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

from mydetector3d.tools.eval_utils import eval_one_epoch
from mydetector3d.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mydetector3d.datasets import build_dataloader
from mydetector3d.models import build_network
from mydetector3d.utils import common_utils

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #"0,1"

#'/home/010796032/3DObject/modelzoo_openpcdet/pointpillar_7728.pth'
#'mydetector3d/tools/cfgs/kitti_models/pointpillar.yaml'
#'/home/010796032/3DObject/3DDepth/output/kitti_models/pointpillar/0413/ckpt/latest_model.pth'

#Second
#'mydetector3d/tools/cfgs/kitti_models/second_multihead.yaml'
#/home/010796032/3DObject/3DDepth/output/kitti_models/second_multihead/0415/ckpt/checkpoint_epoch_256.pth

#cfg_file: 'mydetector3d/tools/cfgs/kitti_models/second.yaml'
#--ckpt: '/home/010796032/3DObject/modelzoo_openpcdet/second_7862.pth'
#pretrained_model: None

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='mydetector3d/tools/cfgs/kitti_models/my3dmodel_multihead.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='0419test', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='/home/010796032/3DObject/3DDepth/output/kitti_models/my3dmodel_multihead/0419/ckpt/checkpoint_epoch_168.pth', help='checkpoint to start from')
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
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem #Returns the substring from the beginning of filename: pointpillar
    #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[-2:-1])# get kitti_models

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    # start evaluation
    eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

from mydetector3d.models.detectors.pointpillar import PointPillar
from mydetector3d.models.detectors.second_net import SECONDNet
from mydetector3d.models.detectors.my3dmodel import My3Dmodel
__modelall__ = {
    #'Detector3DTemplate': Detector3DTemplate,
     'SECONDNet': SECONDNet,
    # 'PartA2Net': PartA2Net,
    # 'PVRCNN': PVRCNN,
     'PointPillar': PointPillar,
     'My3Dmodel': My3Dmodel
}

def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    #Folder format: output/kitti_models(EXP_GROUP_PATH)/pointpillar(TAG)/extra_tag
    #output_dir = cfg.ROOT_DIR / 'output' /  cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    # if not args.eval_all:
    #     num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    #     epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    #     eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    # else:
    #     eval_output_dir = eval_output_dir / 'eval_all_default'

    # if args.eval_tag is not None:
    #     eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    #ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    #model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model_cfg=cfg.MODEL
    model_name=model_cfg.NAME
    num_class=len(cfg.CLASS_NAMES)
    model = __modelall__[model_name](
        model_cfg=model_cfg, num_class=num_class, dataset=test_set
    )
    
    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                    pre_trained_path=args.pretrained_model)
        model.cuda()
        
        # start evaluation
        epoch_id = 256
        ret_dict = eval_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
            result_dir=eval_output_dir
        )
        print(ret_dict)
        #eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()