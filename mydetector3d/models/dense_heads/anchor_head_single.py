import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    """
    Args:
        model_cfg: AnchorHeadSingle configuration
        input_channels: 3
        class_names: ['Car','Pedestrian','Cyclist']
        grid_size: (432, 496, 1), VOXEL_SIZE: [0.16, 0.16, 4] Nx=69.12/0.16=432, Ny=39.68*2/0.16=496, Nz=4/4=1
        point_cloud_range: (0, -39.68, -3, 69.12, 39.68, 1) x range: [0~69.12], y range: [-39.68~39.68], z range: [-3.0, 1] 
        predict_boxes_when_training = False
    Returns:

    """
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        #input_channels: 384, grid_size: (432,493,1)
        self.num_anchors_per_location = sum(self.num_anchors_per_location) #6=sum([2,2,2]), 
        #each point has three anchors, each anchor with two directions (0 and 90 degree)

        self.conv_cls = nn.Conv2d( #384, 6*3 Conv2d(384,18,kernel_size=(1,1),stride=(1,1))
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d( #384, 6*7 Conv2d(384,42,kernel_size=(1,1),stride=(1,1))
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d( #Conv2d(384,12,kernel_size=(1,1),stride=(1,1))
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d'] #[16, 384, 248, 216]

        #each point has 6 anchors, 18=6*3classes
        cls_preds = self.conv_cls(spatial_features_2d) #[16, 18, 248, 216]
        #6 anchors, each anchor has 7 parameters (x,y,z,w,l,h,theta)
        box_preds = self.conv_box(spatial_features_2d) #[16, 42, 248, 216]

        #put class/box in the last dim
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] [16, 248, 216, 18]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] [16, 248, 216, 42]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            #[16,12, 248, 216] 6anchors*2classes(0, 180degree)=12
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() #[16, 248, 216, 12]
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            #targets_dict={'box_cls_labels' [4, #anchors], 'box_reg_targets' [4, #anchors, 7], 'reg_weights' [4, #anchors]}
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'] #4,39,8
            )
            #put matching results to dict
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds #[16, 321408, 3]
            data_dict['batch_box_preds'] = batch_box_preds #[16, 321408, 7]
            data_dict['cls_preds_normalized'] = False

        return data_dict
