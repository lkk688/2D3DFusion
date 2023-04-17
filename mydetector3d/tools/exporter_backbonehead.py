#ref: https://github.com/hova88/OpenPCDet/blob/master/tools/onnx_utils/trans_backbone_multihead.py
import torch
from torch import nn
import numpy as np
#from onnx_backbone_2d import BaseBEVBackbone
#from onnx_dense_head import  AnchorHeadMulti
from mydetector3d.config import cfg, cfg_from_yaml_file

from mydetector3d.models.backbones_2d import BaseBEVBackbone
from mydetector3d.models.dense_heads.anchor_head_template import AnchorHeadTemplate

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #"0,1"

class SingleHead(BaseBEVBackbone):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        num_anchors_per_location,
        code_size,
        rpn_head_cfg=None,
        head_label_indices=None,
        separate_reg_config=None,
    ):
        super().__init__(rpn_head_cfg, input_channels)

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer("head_label_indices", head_label_indices)

        if self.separate_reg_config is not None:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            num_middle_conv = self.separate_reg_config.NUM_MIDDLE_CONV
            num_middle_filter = self.separate_reg_config.NUM_MIDDLE_FILTER
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_conv):
                conv_cls_list.extend(
                    [
                        nn.Conv2d(
                            c_in,
                            num_middle_filter,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_middle_filter),
                        nn.ReLU(),
                    ]
                )
                c_in = num_middle_filter
            conv_cls_list.append(
                nn.Conv2d(
                    c_in,
                    self.num_anchors_per_location * self.num_class,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.conv_cls = nn.Sequential(*conv_cls_list)

            for reg_config in self.separate_reg_config.REG_LIST:
                reg_name, reg_channel = reg_config.split(":")
                reg_channel = int(reg_channel)
                cur_conv_list = []
                c_in = input_channels
                for k in range(num_middle_conv):
                    cur_conv_list.extend(
                        [
                            nn.Conv2d(
                                c_in,
                                num_middle_filter,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_middle_filter),
                            nn.ReLU(),
                        ]
                    )
                    c_in = num_middle_filter

                cur_conv_list.append(
                    nn.Conv2d(
                        c_in,
                        self.num_anchors_per_location * int(reg_channel),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
                code_size_cnt += reg_channel
                self.conv_box[f"conv_{reg_name}"] = nn.Sequential(*cur_conv_list)
                self.conv_box_names.append(f"conv_{reg_name}")

            for m in self.conv_box.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            assert (
                code_size_cnt == code_size
            ), f"Code size does not match: {code_size_cnt}:{code_size}"
        else:
            self.conv_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.num_class,
                kernel_size=1,
            )
            self.conv_box = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.code_size,
                kernel_size=1,
            )

        if self.model_cfg.get("USE_DIRECTION_CLASSIFIER", None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
            )
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get("USE_MULTIHEAD", False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        # spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)

        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = (
                box_preds.view(-1, self.num_anchors_per_location, self.code_size, H, W)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            cls_preds = (
                cls_preds.view(-1, self.num_anchors_per_location, self.num_class, H, W)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = (
                    dir_cls_preds.view(
                        -1,
                        self.num_anchors_per_location,
                        self.model_cfg.NUM_DIR_BINS,
                        H,
                        W,
                    )
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                dir_cls_preds = dir_cls_preds.view(
                    batch_size, -1, self.model_cfg.NUM_DIR_BINS
                )
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        # ret_dict['cls_preds'] = cls_preds
        # ret_dict['box_preds'] = box_preds
        # ret_dict['dir_cls_preds'] = dir_cls_preds

        return cls_preds, box_preds


class AnchorHeadMulti(AnchorHeadTemplate):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        predict_boxes_when_training=True,
        **kwargs,
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training,
        )
        self.model_cfg = model_cfg
        self.separate_multihead = self.model_cfg.get("SEPARATE_MULTIHEAD", False)

        if self.model_cfg.get("SHARED_CONV_NUM_FILTER", None) is not None:
            shared_conv_num_filter = self.model_cfg.SHARED_CONV_NUM_FILTER
            self.shared_conv = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    shared_conv_num_filter,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(shared_conv_num_filter, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            self.shared_conv = None
            shared_conv_num_filter = input_channels
        self.rpn_heads = None
        self.make_multihead(shared_conv_num_filter)

    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg["HEAD_CLS_NAME"])

        for rpn_head_cfg in rpn_head_cfgs:
            num_anchors_per_location = sum(
                [
                    self.num_anchors_per_location[class_names.index(head_cls)]
                    for head_cls in rpn_head_cfg["HEAD_CLS_NAME"]
                ]
            )
            head_label_indices = torch.from_numpy(
                np.array(
                    [
                        self.class_names.index(cur_name) + 1
                        for cur_name in rpn_head_cfg["HEAD_CLS_NAME"]
                    ]
                )
            )

            rpn_head = SingleHead(
                self.model_cfg,
                input_channels,
                len(rpn_head_cfg["HEAD_CLS_NAME"])
                if self.separate_multihead
                else self.num_class,
                num_anchors_per_location,
                self.box_coder.code_size,
                rpn_head_cfg,
                head_label_indices=head_label_indices,
                separate_reg_config=self.model_cfg.get("SEPARATE_REG_CONFIG", None),
            )
            rpn_heads.append(rpn_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, spatial_features_2d):
        # spatial_features_2d = data_dict['spatial_features_2d']
        data_dict = {}
        if self.shared_conv is not None:
            spatial_features_2d = self.shared_conv(spatial_features_2d)

        # ret_dicts = []
        cls_preds = []
        box_preds = []
        for rpn_head in self.rpn_heads:
            # ret_dicts.append(rpn_head(spatial_features_2d))
            cls_pred, box_pred = rpn_head(spatial_features_2d)
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        # cls_preds = [ret_dict['cls_preds'] for ret_dict in ret_dicts]
        # box_preds = [ret_dict['box_preds'] for ret_dict in ret_dicts]
        # ret = {
        #     'cls_preds': cls_preds if self.separate_multihead else torch.cat(cls_preds, dim=1),
        #     'box_preds': box_preds if self.separate_multihead else torch.cat(box_preds, dim=1),
        # }

        # if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
        #     dir_cls_preds = [ret_dict['dir_cls_preds'] for ret_dict in ret_dicts]
        #     ret['dir_cls_preds'] = dir_cls_preds if self.separate_multihead else torch.cat(dir_cls_preds, dim=1)

        # self.forward_ret_dict.update(ret)

        # if self.training:
        #     targets_dict = self.assign_targets(
        #         gt_boxes=data_dict['gt_boxes']
        #     )
        #     self.forward_ret_dict.update(targets_dict)

        # if not self.training or self.predict_boxes_when_training:
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=1, cls_preds=cls_preds, box_preds=box_preds
        )

        #     if isinstance(batch_cls_preds, list):
        #         multihead_label_mapping = []
        #         for idx in range(len(batch_cls_preds)):
        #             multihead_label_mapping.append(self.rpn_heads[idx].head_label_indices)

        #         data_dict['multihead_label_mapping'] = multihead_label_mapping

        #     data_dict['cls_preds'] = ret['cls_preds']
        #     data_dict['box_preds'] = ret['box_preds']

        #     data_dict['batch_cls_preds'] = batch_cls_preds
        #     data_dict['batch_box_preds'] = batch_box_preds
        #     data_dict['cls_preds_normalized'] = False

        return batch_cls_preds, batch_box_preds  # cls_preds, box_preds

    def get_cls_layer_loss(self):
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if "pos_cls_weight" in loss_weights:
            pos_cls_weight = loss_weights["pos_cls_weight"]
            neg_cls_weight = loss_weights["neg_cls_weight"]
        else:
            pos_cls_weight = neg_cls_weight = 1.0

        cls_preds = self.forward_ret_dict["cls_preds"]
        box_cls_labels = self.forward_ret_dict["box_cls_labels"]
        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]
        batch_size = int(cls_preds[0].shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0 * neg_cls_weight

        cls_weights = (negative_cls_weights + pos_cls_weight * positives).float()

        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()

        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            self.num_class + 1,
            dtype=cls_preds[0].dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        start_idx = c_idx = 0
        cls_losses = 0

        for idx, cls_pred in enumerate(cls_preds):
            cur_num_class = self.rpn_heads[idx].num_class
            cls_pred = cls_pred.view(batch_size, -1, cur_num_class)
            if self.separate_multihead:
                one_hot_target = one_hot_targets[
                    :,
                    start_idx : start_idx + cls_pred.shape[1],
                    c_idx : c_idx + cur_num_class,
                ]
                c_idx += cur_num_class
            else:
                one_hot_target = one_hot_targets[
                    :, start_idx : start_idx + cls_pred.shape[1]
                ]
            cls_weight = cls_weights[:, start_idx : start_idx + cls_pred.shape[1]]
            cls_loss_src = self.cls_loss_func(
                cls_pred, one_hot_target, weights=cls_weight
            )  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss = cls_loss * loss_weights["cls_weight"]
            cls_losses += cls_loss
            start_idx += cls_pred.shape[1]
        assert start_idx == one_hot_targets.shape[1]
        tb_dict = {"rpn_loss_cls": cls_losses.item()}
        return cls_losses, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict["box_preds"]
        box_dir_cls_preds = self.forward_ret_dict.get("dir_cls_preds", None)
        box_reg_targets = self.forward_ret_dict["box_reg_targets"]
        box_cls_labels = self.forward_ret_dict["box_cls_labels"]

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [
                        anchor.permute(3, 4, 0, 1, 2, 5)
                        .contiguous()
                        .view(-1, anchor.shape[-1])
                        for anchor in self.anchors
                    ],
                    dim=0,
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        start_idx = 0
        box_losses = 0
        tb_dict = {}
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(
                batch_size,
                -1,
                box_pred.shape[-1] // self.num_anchors_per_location
                if not self.use_multihead
                else box_pred.shape[-1],
            )
            box_reg_target = box_reg_targets[
                :, start_idx : start_idx + box_pred.shape[1]
            ]
            reg_weight = reg_weights[:, start_idx : start_idx + box_pred.shape[1]]
            # sin(a - b) = sinacosb-cosasinb
            if box_dir_cls_preds is not None:
                box_pred_sin, reg_target_sin = self.add_sin_difference(
                    box_pred, box_reg_target
                )
                loc_loss_src = self.reg_loss_func(
                    box_pred_sin, reg_target_sin, weights=reg_weight
                )  # [N, M]
            else:
                loc_loss_src = self.reg_loss_func(
                    box_pred, box_reg_target, weights=reg_weight
                )  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS["loc_weight"]
            box_losses += loc_loss
            tb_dict["rpn_loss_loc"] = tb_dict.get("rpn_loss_loc", 0) + loc_loss.item()

            if box_dir_cls_preds is not None:
                if not isinstance(box_dir_cls_preds, list):
                    box_dir_cls_preds = [box_dir_cls_preds]
                dir_targets = self.get_direction_target(
                    anchors,
                    box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS,
                )
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(
                    batch_size, -1, self.model_cfg.NUM_DIR_BINS
                )
                weights = positives.type_as(dir_logit)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)

                weight = weights[:, start_idx : start_idx + box_pred.shape[1]]
                dir_target = dir_targets[:, start_idx : start_idx + box_pred.shape[1]]
                dir_loss = self.dir_loss_func(dir_logit, dir_target, weights=weight)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = (
                    dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS["dir_weight"]
                )
                box_losses += dir_loss
                tb_dict["rpn_loss_dir"] = (
                    tb_dict.get("rpn_loss_dir", 0) + dir_loss.item()
                )
            start_idx += box_pred.shape[1]
        return box_losses, tb_dict

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get("LAYER_NUMS", None) is not None:
            assert (
                len(self.model_cfg.LAYER_NUMS)
                == len(self.model_cfg.LAYER_STRIDES)
                == len(self.model_cfg.NUM_FILTERS)
            )
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get("UPSAMPLE_STRIDES", None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(
                self.model_cfg.NUM_UPSAMPLE_FILTERS
            )
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, spatial_features):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # data_dict['spatial_features_2d'] = x

        return x
    

class backbone(nn.Module):
    def __init__(self, cfg , gridx , gridy):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
        self.dense_head =  AnchorHeadMulti(
            model_cfg=cfg.MODEL.DENSE_HEAD,
            input_channels=384,
            num_class=len(cfg.CLASS_NAMES),
            class_names=cfg.CLASS_NAMES,
            grid_size=np.array([gridx , gridy , 1]),
            point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            predict_boxes_when_training=False)

    def forward(self, spatial_features):
        x = self.backbone_2d(spatial_features)
        batch_cls_preds, batch_box_preds = self.dense_head.forward(x)

        return batch_cls_preds, batch_box_preds


def build_backbone_multihead(ckpt , cfg ):

    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    grid_size = (pc_range[3:] - pc_range[:3]) /voxel_size
    gridx = grid_size[0].astype(np.int) #432
    gridy = grid_size[1].astype(np.int) #496
    model = backbone(cfg , gridx ,gridy)
    model.to('cuda').eval()

    checkpoint = torch.load(ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key] = checkpoint["model_state"][key]
        if "dense_head" in key:
            dicts[key] = checkpoint["model_state"][key]
    model.load_state_dict(dicts)

    dummy_input = torch.ones(1, 64, gridx, gridy).cuda()
    return model , dummy_input

if __name__ == "__main__":
    import numpy as np 
    #cfg_file = 'mydetector3d/tools/cfgs/kitti_models/pointpillar.yaml' #cfg_file = '/path/to/cbgs_pp_multihead.yaml'
    #cfg_file = 'mydetector3d/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
    #filename_mh = '/home/010796032/3DObject/modelzoo_openpcdet/pp_multihead_nds5823_updated.pth'#pointpillar_7728.pth' # filename_mh = "./output/pp_multihead_nds5823_updated.pth"
    
    cfg_file = 'mydetector3d/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml' #'/path/to/cbgs_pp_multihead.yaml'
    filename_mh = '/home/010796032/3DObject/modelzoo_openpcdet/pp_multihead_nds5823_updated.pth'
    cfg_from_yaml_file(cfg_file, cfg)
    model , dummy_input = build_backbone_multihead(filename_mh , cfg )

    export_onnx_file = "./output/cbgs_pp_multihead_backbone.onnx"
    model.eval().cuda()
    #Exporting the operator 'aten::atan2' to ONNX opset version 12 is not supported.
    #Change atan2->atan in utils/box_coder_utils.py
    torch.onnx.export(model,
                      dummy_input,
                      export_onnx_file,
                      opset_version=12,
                      verbose=True,
                      do_constant_folding=True) # 输出名