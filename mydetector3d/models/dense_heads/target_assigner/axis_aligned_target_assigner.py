import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG #assign box to target configuration
        self.box_coder = box_coder #utils.box_coder_utils.ResidualCoder, size=7 (xyz, wlh, theta)
        self.match_height = match_height # False, no need to match height as no object is overlapped in z-axis
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None #none
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE #512
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES #False
        self.matched_thresholds = {} #positive sample threshold: {'Car': 0.6, 'Pedestrian': 0.5, 'Cyclist': 0.5}
        self.unmatched_thresholds = {} #negative sample threshold: {'Car': 0.45, 'Pedestrian': 0.35, 'Cyclist': 0.35}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold'] #0.6
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold'] #0.45

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]   [1, 248, 216, 1, 2, 7]*3
            gt_boxes: (B, M, 8)     [16, 45, 8], 8 is (x,y,z,w,l,h,theta,class)
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1] #[16, 45]
        gt_boxes = gt_boxes_with_classes[:, :, :-1] #[16, 45, 7] remove 1 (class) from 8->7 (x,y,z,w,l,h,theta)
        for k in range(batch_size): #match each frame to anchor's foreground and background
            cur_gt = gt_boxes[k] #[45, 7] [44,7] get gt_boxes in current frame (some GT boxes are 0 in pre-process to match max GT size)
            cnt = cur_gt.__len__() - 1 #44 43
            while cnt > 0 and cur_gt[cnt].sum() == 0: #find the last box with no zero, i.e., actual number of box
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1] #[34, 7] get the true gt_boxes without 0
            cur_gt_classes = gt_classes[k][:cnt + 1].int() #34 get the true gt+class without 0

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors): #for each class
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name) #select 'Car' in all gt [44]
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead: #anchor [1, 200, 176, 1, 2, 7]->[1 2 1 200 176 7]->[70400, 7]
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask] #18 Cars
                else:
                    feature_map_size = anchors.shape[:3] #[1, 248, 216]
                    anchors = anchors.view(-1, anchors.shape[-1]) #[107136, 7]
                    selected_classes = cur_gt_classes[mask] #13
                #single class
                single_target = self.assign_targets_single(
                    anchors, #all anchors
                    cur_gt[mask], #true gt_boxes (num_gtbox,7)
                    gt_classes=selected_classes, #18 array
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets']) #[211200, 7]
            cls_labels.append(target_dict['box_cls_labels']) #[211200]
            reg_weights.append(target_dict['reg_weights']) #[211200] regression weight (foreground is 1, background=0)

        bbox_targets = torch.stack(bbox_targets, dim=0) #[16, 211200, 7] for the whole batch

        cls_labels = torch.stack(cls_labels, dim=0) #[16, 211200]
        reg_weights = torch.stack(reg_weights, dim=0) #[16, 211200]
        all_targets_dict = {
            'box_cls_labels': cls_labels, #[16, 211200]
            'box_reg_targets': bbox_targets, #[16, 211200, 7]
            'reg_weights': reg_weights #[16, 211200] regression weight (foreground is 1, background=0)

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        #process for each class, e.g., 'Car'
        num_anchors = anchors.shape[0] #107136
        num_gt = gt_boxes.shape[0] #13
        #initialize labels and gt_ides for each anchor to -1, Loss will not calcuate -1, background=0
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 #[num_anchor] size
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7]) #if not match height, only use 2D box in bev
            #anchor_by_gt_overlap:[70400, 18] iou for each anchor and gt_box (jaccard index), most value =0
            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1) #[70400] get anchor's max IOU among all gt boxes
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax] #iou value for most matched gt box

            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0) #[70400, 18]->[18] max in dim=0: for each gt, get the most matched anchor index
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] # IOU value for most matched anchor
            empty_gt_mask = gt_to_anchor_max == 0 #which gt_box did not get matched anchor
            gt_to_anchor_max[empty_gt_mask] = -1 #set these unmatched gt's IOU=-1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] #[21] number of multiple best matching IOU
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] #[21] matched with which gt index
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] #[70400] put gt index into anchors' label
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() #[70400] put gt index into anchors' gt_ids

            pos_inds = anchor_to_gt_max >= matched_threshold #iou of most matched anchor > threshold [70400]
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] #get the gt index [85]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh] #[70400] put gt class to the anchor's label
            gt_ids[pos_inds] = gt_inds_over_thresh.int() #put the gt index to gt_ids
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] #[70179] find the index for background
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0] #[85] foreground anchor index

        if self.pos_fraction is not None: #sample the foreground, here is none
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0 #if no gt, all labels are background (0)
            else:
                labels[bg_inds] = 0 #background set to 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] #[70400] set classes to anchor

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size)) #[70400, 7]
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :] #[85, 7] foreground gt box for each anchor
            fg_anchors = anchors[fg_inds, :] #get foreground anchor [85, 7]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors) #[70400, 7] encoding for 7 parameters (delta_x,y,z, delta_xlh, delta_theta)

        reg_weights = anchors.new_zeros((num_anchors,)) #initialize regression weights [70400]

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0 #set the weight of foreground anchor =1

        ret_dict = {
            'box_cls_labels': labels, #[70400]
            'box_reg_targets': bbox_targets, #[70400, 7]
            'reg_weights': reg_weights, #[70400]
        }
        return ret_dict
