import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']#[254904, 5, 4], [254904], max_points_per_voxel=5
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) #points_mean for 5 points in one voxel: [254904, 4]
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features) #prevent 0
        points_mean = points_mean / normalizer #divide number of points
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
