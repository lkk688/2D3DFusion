from .detector3d_template import Detector3DTemplate
from mydetector3d.models.sub_modules.compress import NaiveCompressor 
#modified based on PointPillar
class My3Dmodelv2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        #self.module_list = self.build_networks()
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, #4
            'num_point_features': self.dataset.point_feature_encoder.num_point_features, #4
            'grid_size': self.dataset.grid_size, #（432,496, 1）
            'point_cloud_range': self.dataset.point_cloud_range, #[0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
            'voxel_size': self.dataset.voxel_size, #[0.16, 0.16, 4]
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        print("Num point features initial", model_info_dict['num_point_features']) #4
        vfe_module, model_info_dict = self.build_vfe(model_info_dict=model_info_dict) #PillarVFE
        self.add_module('vfe', vfe_module)#nn.module add_module
        print("Num point features after VFE", model_info_dict['num_point_features']) #64 (feature10->64)
        #model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        #model_info_dict['module_list'].append(vfe_module)

        #backbone_3d_module, model_info_dict = self.build_backbone_3d(model_info_dict=model_info_dict) #None for PointPillar

        map_to_bev_module, model_info_dict = self.build_map_to_bev_module(model_info_dict=model_info_dict) #PointPillarScatter
        self.add_module('map_to_bev_module', map_to_bev_module)#nn.module add_module
        print("num_bev_features features after BEV", model_info_dict['num_bev_features']) #64
        #model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features

        #pfe_module, model_info_dict = self.build_pfe(model_info_dict=model_info_dict) #None for PointPillar

        backbone_2d_module, model_info_dict = self.build_backbone_2d(model_info_dict=model_info_dict) #BaseBEVBackbone
        self.add_module('backbone_2d', backbone_2d_module)#nn.module add_module
        print("num_bev_features features after backbone2d", model_info_dict['num_bev_features']) #384
        #model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features

        compress_raito =2
        compressor = NaiveCompressor(384, compress_raito)
        model_info_dict['module_list'].append(compressor)
        self.add_module('compressor_module', compressor)
        #data_dict['spatial_features_2d']


        dense_head_module, model_info_dict = self.build_dense_head(model_info_dict=model_info_dict) #AnchorHeadSingle
        self.add_module('dense_head', dense_head_module)#nn.module add_module
        print("Num point features after dense_head", model_info_dict['num_point_features'])

        #point_head_module, model_info_dict = self.build_point_head(model_info_dict=model_info_dict) # None for PointPillar
        #point_head_module, model_info_dict = self.build_roi_head(model_info_dict=model_info_dict) # None for PointPillar

        self.module_list =  model_info_dict['module_list'] #PillarVEF, PointPillarScatter, BaseBEVBackbone, AnchorHeadSingle
        # self.module_topology = [
        #     'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
        #     'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        # ]
    
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts #batch size array of dicts, each dict contains 'pred_boxes': [11, 7], 'pred_scores', and 'pred_labels'

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict