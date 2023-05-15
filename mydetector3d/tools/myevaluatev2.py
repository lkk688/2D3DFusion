from scipy.fft import fft, ifft, fftfreq, fftshift #solve the ImportError: /cm/local/apps/gcc/11.2.0/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
import argparse
import datetime
#import glob
import os
import re
import time
from pathlib import Path
import pickle
import tqdm
import numpy as np
import torch
from easydict import EasyDict
import copy

def check_gpus():
    import nvidia_smi #pip install nvidia-ml-py3
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(GB total), {} (GB free), {} (GB used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/1024**3, info.free/1024**3, info.used/1024**3))

    nvidia_smi.nvmlShutdown()

def torch_setgpu(gpuid):
    if torch.cuda.is_available():
        print('current device before setup:', torch.cuda.current_device())
        total_gpus = torch.cuda.device_count()
        if gpuid is not None and gpuid < total_gpus:
            print("Use GPU: {} for training".format(gpuid))
        else:
            gpuid = 0
            print("GPUID larger than number of GPUS: {}, Use GPU: 0 for training".format(total_gpus))
        torch.cuda.set_device(gpuid) #model.cuda(args.gpuid)
        device = torch.device('cuda:{}'.format(gpuid))
        print('current device after setup:', torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    return device

from mydetector3d.config import cfg_from_yaml_file, log_config_to_file #, cfg
from mydetector3d.utils import common_utils
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
from mydetector3d.datasets.kitti.waymokitti_simpledataset import WaymoKittiDataset
from mydetector3d.datasets.kitti.dairkitti_dataset import DairKittiDataset
from mydetector3d.datasets.waymo.waymo_dataset import WaymoDataset
from torch.utils.data import DataLoader
__datasetall__ = {
    'KittiDataset': KittiDataset,
    'WaymoKittiDataset': WaymoKittiDataset,
    'waymokitti_dataset': WaymoKittiDataset,
    'WaymoDataset': WaymoDataset,
    'DairKittiDataset': DairKittiDataset
}

#newly created
def load_data_to_device(batch_dict, device):
    if type(batch_dict) is dict:
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            # elif key in ['images']:
            #     batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            # elif key in ['images']:
            #     batch_dict[key] = kornia.image_to_tensor(val).float().to(device).contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = batch_dict[key] = torch.from_numpy(val).int().to(device) #torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().to(device) #torch.from_numpy(val).float().cuda()
    # else:
    #     batch_dict = batch_dict.to(device)

#'/data/cmpe249-fa22/Mymodels/waymokitti_models/second/0502/ckpt/checkpoint_epoch_128.pth'

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='mydetector3d/tools/cfgs/dairkitti_models/my3dmodel.yaml', help='specify the model config')
    parser.add_argument('--dataset_cfg_file', type=str, default=None, help='specify the dataset config')
    #parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default='/data/cmpe249-fa22/Mymodels/dairkitti_models/my3dmodel/0513/ckpt/checkpoint_epoch_128.pth', help='checkpoint to evaluate')
    parser.add_argument('--outputpath', type=str, default='/data/cmpe249-fa22/Mymodels/', help='output path')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--save_to_file', default=True, help='')
    parser.add_argument('--kittiformat', default=True, help='')
    parser.add_argument('--eval_only', default=False, help='') #When detection result is available, set to True and just run the evaluation
    parser.add_argument('--savebatchidx', type=int, default=1, help='Save one batch data to pkl for visualization')
    parser.add_argument('--infer_time', default=True, help='calculate inference latency') #action='store_true' true if specified

    args = parser.parse_args()

    cfg = EasyDict()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.modelnamepath = Path(args.cfg_file).stem #Returns the substring from the beginning of filename: pointpillar
    
    ckptsplits = args.ckpt.split('/')  #[-2:-1]
    cfg.datasetname = ckptsplits[-5] #Path(args.dataset_cfg_file).stem
    epoch =Path(ckptsplits[-1]).stem #remove .pth checkpoint_epoch_128.pth
    epoch =epoch.split('_')[-1] #get epoch number
    args.savename = cfg.datasetname + '_' + cfg.modelnamepath + '_epoch' + epoch


    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU #4

    args.output_dir = Path(args.outputpath) / 'eval' / args.savename
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.eval_output_dir = args.output_dir / 'txtresults' #'eval'
    args.eval_output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(1024)

    return args, cfg

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def load_weights_from_file(model, filename, device):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    print('==> Loading parameters from checkpoint %s to %s' % (filename, device))
    #loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=device) #loc = 'cuda:{}'.format(args.gpu)
    model_state_disk = checkpoint['model_state']

    state_dict, update_model_state = model._load_state_dict(model_state_disk, strict=False)

    for key in state_dict:
        if key not in update_model_state:
            print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    print('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

def main():
    args, cfg = parse_config()

    log_file = args.output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    # if args.infer_time:
    #     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    check_gpus()
    device = torch_setgpu(args.gpuid)

    # Create dataset
    datasetcfg = EasyDict()
    if  args.dataset_cfg_file is not None:
        #dataset_cfg = args.dataset_cfg_file #cfg.DATA_CONFIG
        cfg_from_yaml_file( args.dataset_cfg_file, datasetcfg)
    else: 
        datasetcfg=cfg.DATA_CONFIG #dict
        #cfg_from_yaml_file(cfg.DATA_CONFIG, datasetcfg) #using the default DATA_CONFIG in the model cfg file
    class_names=cfg.CLASS_NAMES #classnames from the model
    dataset = __datasetall__[datasetcfg.DATASET](
        dataset_cfg=datasetcfg,
        class_names=class_names,
        root_path=None,
        training=False,
        logger=logger,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
        shuffle=None, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=None, timeout=0, worker_init_fn=None)
    
    if not args.eval_only:
        #Build Model
        model_cfg=cfg.MODEL
        model_name=model_cfg.NAME
        num_class=len(cfg.CLASS_NAMES)
        model = __modelall__[model_name](
            model_cfg=model_cfg, num_class=num_class, dataset=dataset
        )
        # load checkpoint
        # model.load_params_from_file(filename=args.ckpt, logger=None, to_cpu=False, 
        #                             pre_trained_path=None)
        load_weights_from_file(model, args.ckpt, device)
        #model.cuda()
        model.to(device)
        model.eval()

        det_annos, ret_dicts, ret_dict = rundetection(dataloader, model, device, cfg, args, args.eval_output_dir)
        resultfile=args.output_dir / 'result.pkl'
        with open(resultfile, 'wb') as f:
            pickle.dump(det_annos, f)
        with open(args.output_dir / 'ret_dicts.pkl', 'wb') as f:
            pickle.dump(ret_dicts, f)
        print("Finished detection:", ret_dict)
    else:
        #load previous saved pkl
        resultfile=args.output_dir / 'result.pkl'
        f = open(resultfile, 'rb')   # if only use 'r' for reading; it will show error: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
        det_annos = pickle.load(f)         # load file content as mydict
        f.close()

    
    result_str, result_dict =runevaluation(dataset, det_annos, class_names, args.eval_output_dir, args.kittiformat)
    print(result_str)
    print(result_dict)


def rundetection(dataloader, model, device, cfg, args, eval_output_dir):
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
        ret_dicts = []

        if getattr(args, 'infer_time', False):
            start_iter = int(len(dataloader) * 0.1)
            infer_time_meter = common_utils.AverageMeter()
        
        start_time = time.time()

        # #batch_dict data:
        # 'gt_boxes': (16, 16, 8), 16: batch size, 16: number of boxes (many are zeros), 8: boxes value
        # 'points': (302730, 5): 5: add 0 in the left of 4 point features (xyzr)
        # Voxels: (89196, 32, 4) 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
        # Voxel_coords: (89196, 4) (batch_index,z,y,x) added batch_index in dataset.collate_batch
        # Voxel_num_points: (89196,)
        for i, batch_dict in enumerate(dataloader):
            #load_data_to_gpu(batch_dict)
            load_data_to_device(batch_dict, device) #dict cannot use .to(device)

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
            
            #save more information to ret_dict
            ret_dict['infer_time'] = infer_time_meter.val
            ret_dict['pred_dicts'] = pred_dicts
            ret_dict['gt_boxes'] = batch_dict['gt_boxes']
            ret_dicts.append(ret_dict)

            #convert to Kitti format, save result to txt file
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=eval_output_dir if args.save_to_file else None
            )#batch_size array, each dict is the Kitti annotation-like format dict (2D box is converted from 3D pred box)
            #annos batch size=4 dict array, each contains 'name' array(335), 'score(335)', 'boxes_lidar(335,7)', 'pred_labels(335)'
            det_annos += annos #annos array: batchsize(16) pred_dict in each batch; det_annos array: all objects in all frames in the dataset
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

            if args.savebatchidx is not None and i==args.savebatchidx:
                #save the current batch data for later evaluation
                load_data_to_device(batch_dict,'cpu')
                load_data_to_device(pred_dicts,'cpu')
                save_dict = {}
                save_dict['idx']=i
                save_dict['ckpt']=args.ckpt
                save_dict['cfg_file']=args.cfg_file
                save_dict['datasetname']=args.dataset_cfg_file
                save_dict['batch_dict']=batch_dict #batch_dict#.numpy()
                save_dict['pred_dicts']=pred_dicts #pred_dicts#.numpy()
                save_dict['annos']=annos#.numpy()
                save_dict['infer_time']=infer_time_meter.val
                resultfile=args.savename + '_frame_%s.pkl' % str(i)
                with open(args.output_dir / resultfile, 'wb') as f:
                    pickle.dump(save_dict, f)
        progress_bar.close()

        ret_dict = {}
        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            print('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            print('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos: #each frame's results
            total_pred_objects += anno['name'].__len__()
        print('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    ret_dict['infer_time']=infer_time_meter.avg
    ret_dict['total_pred_objects']=total_pred_objects
    ret_dict['total_annos']=len(det_annos)
    return det_annos, ret_dicts, ret_dict

def runevaluation(dataset, det_annos, class_names, final_output_dir, kittiformat=False):
        # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC, #kitti
    #     output_path=final_output_dir
    # )
    #from .kitti_object_eval_python import eval as kitti_eval
    from mydetector3d.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    from mydetector3d.datasets.kitti.kitti_utils import transform_annotations_to_kitti_format
    

    if hasattr(dataset, "kitti_infos"):
        datainfo=dataset.kitti_infos
    elif hasattr(dataset, "infos"):
        datainfo=dataset.infos #waymodataset has different name
    else:
        print("datainfo not exist")
        return

    #remove objects not in the Kitti classes list
    eval_det_annos = [] #copy.deepcopy(det_annos) #'boxes_lidar'
    eval_gt_annos = [] #[copy.deepcopy(info['annos']) for info in datainfo]
    total_framenum= len(datainfo) #size of eval_det_annos should be the same to eval_gt_annos
    det_emptyframe=[]
    gt_emptyframe=[]
    kitticlass_names=['Car', 'Pedestrian', 'Cyclist'] #['Car', 'Pedestrian', 'Cyclist', 'Other']
    for k in range(total_framenum):
        info = datainfo[k]
        det_annotation = det_annos[k]
        annotation=info['annos']
        names_array = annotation['name']
        inds = [i for i, x in enumerate(names_array) if x in kitticlass_names]
        if len(inds)>0: #remove frames with no objects and not in kitticlass_names list
            inds = np.array(inds, dtype=np.int64)
            for key in annotation.keys():
                #print(key)
                #print(annotation[key].shape)
                if annotation[key].ndim==2:
                    annotation[key]=annotation[key][inds,:]
                elif annotation[key].ndim==1:
                    annotation[key]=annotation[key][inds]
                #print(annotation[key].shape)
            eval_gt_annos.append(annotation)
            eval_det_annos.append(det_annotation)
        else:
            det_emptyframe.append(det_annotation)
            gt_emptyframe.append(annotation)

    
    if kittiformat is not True:
        map_name_to_kitti = {
                    'Vehicle': 'Car',
                    'Pedestrian': 'Pedestrian',
                    'Cyclist': 'Cyclist',
                    'Sign': 'Sign',
                    'Car': 'Car'
                }
        # newclassnames = []
        # for classname in class_names:
        #     if classname in map_name_to_kitti:
        #         newclassname = map_name_to_kitti[classname]
        #         newclassnames.append(newclassname)
        class_names = [map_name_to_kitti[x] for x in class_names]
        

        transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        transform_annotations_to_kitti_format(
                    eval_gt_annos, map_name_to_kitti=map_name_to_kitti, info_with_fakelidar = False)

    
    # for det_anno in det_annos:
    #     #for each frame, replace 'Vehicle' into Kitti's name "Car"
    #     names=det_anno['name'] #array of all names
    #     det_anno['name']=[sub.replace('Vehicle','Car') for sub in names]
    #     eval_det_annos.append(det_anno)
    #     if 'alpha' not in det_anno.keys():
    #         det_anno['alpha']=np.array([-10, -10]) #waymo dataset do not have alpha result
    
    class_to_name = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
            3: 'Other', #'Sign', #'Van',
            4: 'Person_sitting',
            5: 'Truck'
        }
    result_str, result_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, kitticlass_names, class_to_name)
    text_file = open(final_output_dir / "evalresult_str", "w")
    text_file.write(result_str)
    text_file.close()

    resultfile=final_output_dir / 'evalresult_dict.pkl'
    with open(resultfile, 'wb') as f:
        pickle.dump(result_dict, f)
    #ret_dict.update(result_dict)

    print('Result is saved to %s' % final_output_dir)
    print('****************Evaluation done.*****************')
    return result_str, result_dict


if __name__ == '__main__':
    main()