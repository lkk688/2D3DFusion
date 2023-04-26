import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'] #[116106, 128] spacial shape:[2, 200, 176]
        spatial_features = encoded_spconv_tensor.dense() #[16, 128, 2, 200, 176]
        N, C, D, H, W = spatial_features.shape #C=128, D=2, H=200, W=176
        spatial_features = spatial_features.view(N, C * D, H, W) #[16, 256, 200, 176]
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride'] #8
        return batch_dict