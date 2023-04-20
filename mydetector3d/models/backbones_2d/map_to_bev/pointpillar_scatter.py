import torch
import torch.nn as nn


class PointPillarScatter(nn.Module): #return pillar to the orignal location based on the index
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES #64
        self.nx, self.ny, self.nz = grid_size # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] #[89196, 64] [89196, 4]
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size): #16
            spatial_feature = torch.zeros( #[64, 214272]
                self.num_bev_features,
                self.nz * self.nx * self.ny, #1x432x496=214272
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx # (batch_index,z,y,x) match batch_index 89196
            this_coords = coords[batch_mask, :] #[3380, 4]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # x+y*nx+z [3380]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :] #[3380, 64]
            pillars = pillars.t() #[64, 3380]
            spatial_feature[:, indices] = pillars # many of them are zero
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) #16*[64, 214272] =>[16, 64, 214272]
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features #[16, 64, 496, 432]
        return batch_dict