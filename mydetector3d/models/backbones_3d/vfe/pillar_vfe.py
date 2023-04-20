import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer #true
        self.use_norm = use_norm #true
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm: #in:10, out:64
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs): #	[14290, 32, 10] input, 32 is max_points_per_voxel, 10 is feature
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part #1 part, 89196/50000
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0) #part1: [50000, 32, 64], part2: [39196, 32, 64] =>[89196, 32, 64]
        else:
            x = self.linear(inputs) #[14290, 32, 64]
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x #[89196, 32, 64]
        torch.backends.cudnn.enabled = True
        x = F.relu(x) #[89196, 32, 64]
        x_max = torch.max(x, dim=1, keepdim=True)[0] #[14290, 1, 64]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM #True
        self.with_distance = self.model_cfg.WITH_DISTANCE #False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        #10 features:  xyzintensity (4)  points_mean(3) relative to center (3)
        #or 7 features
        if self.with_distance:#using distance as feature sqrt(x^2+y^2+z^2)
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS #[64]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) #[10, 64]

        pfn_layers = []
        for i in range(len(num_filters) - 1): #len=1
            in_filters = num_filters[i] #10
            out_filters = num_filters[i + 1] #64
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers) #only one layer

        self.voxel_x = voxel_size[0] #0.16m
        self.voxel_y = voxel_size[1] #0.16m
        self.voxel_z = voxel_size[2] #4m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #0.16/2+0 = 0.08m
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #0.16/2+(-39.68)=-39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #4/2 + (-3) = -1

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1) #20
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape) #[1, 32]
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator #[89196, 32]

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict:
            points:(97687,5)
            frame_id:(1,) ->array([0])
            gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
            use_lead_xyz:(4,) --> (1,1,1,1)
            voxels_features:(xx,32,4) --> 32 is max_points_per_voxel 4 is feature(x,y,z,intensity)
            voxel_coords:(xx,4) --> (batch_index,z,y,x) added batch_index in dataset.collate_batch
            voxel_num_points:(89196,)
            image_shape:(4,2) [[375 1242],[374 1238],[375 1242],[375 1242]]
            batch_size:4
        """
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        #mean xyz for every points in each voxel [89196,1,3] -> [xxx,:,3](dim=1 sum of 32 points xyz)/[xxx,1,1]

        f_cluster = voxel_features[:, :, :3] - points_mean #points xyz become relative to the voxel mean: xyz-mean [xx,32,3]

        f_center = torch.zeros_like(voxel_features[:, :, :3]) #[xx,32,3] each point distance to the voxel center
        # coords is grid number * voxel size + grid offset = real coordinate
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center] #10: xyzintensity feature (4) + relative xyz (3) + grid center real coordinate (3)
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center] #7: intensity feature (1) + relative xyz (3) + grid center real coordinate (3)

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True) #square distance
            features.append(points_dist)
        features = torch.cat(features, dim=-1) #[8599, 32, 4],[8599, 32, 3],[8599, 32, 3] cat in last dimension 10=4+3+3
        #[xxx,32,10], 10 features:  xyzintensity (4)  points_mean(3) relative to center (3)

        voxel_count = features.shape[1] #32
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0) # (89196,32)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) # (89196,32,1)
        features *= mask # [89196, 32, 10]*[89196, 32, 1] some voxel is not full, add 0
        for pfn in self.pfn_layers:#only one layer
            features = pfn(features) #[89196, 32, 10] 32 is max_points_per_voxel->[89196, 1, 64] max in each voxel
        features = features.squeeze() #[89196, 1, 64]->[89196, 64], every pillar get 64 feature
        batch_dict['pillar_features'] = features
        return batch_dict
