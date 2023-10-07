import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range #(0, -39.68, -3, 69.12, 39.68, 1)
        #get anchor size from configuration, three sizes for ['Car','Pedestrian','Cyclist']
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        #get anchor rotation degree in arc degree [0, 1.57] (0, 90degree)
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        #anchor bottom heights in z-axis
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]#False

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes) #3

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        # for loop for three classes
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            #2=2x1x1
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                #map anchor back to the original point cloud, (69.12-0)/(248-1)=0.32m
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                #(39.68-(-39.68))/(248-1)=0.32m
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                #each point offset to the top/left corner
                x_offset, y_offset = 0, 0

            #all x coordinates from [0, 69.12m], 216points
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            #all y coordinates from [0, 79.36], 248points
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            #[-1.78]
            z_shifts = x_shifts.new_tensor(anchor_height)

            #num_anchor_size=1, num_anchor_rotation=2
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation) #[0, 1.57]
            anchor_size = x_shifts.new_tensor(anchor_size) 
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid] (216,248,1)

            #each anchor location
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]-->[216,248,1,3], 3 means zyx
            #(216,248,1,3)->(216,248,1,1,3)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            #(1,1,1,1,3)-->(216,248,1,1,3)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            #add anchor size info to anchor (216,248,1,1,6), 6 means: z,y,x,l,w,h
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            
            #add a new dimension in second last, repreat this dimension 
            # (216,248,1,1,2,6) means z, y, x, #anchor_size, #rotation, 6(z,y,x,l,w,h)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            #(216,248,1,1,2,1) anchor rotation for two anchors with different rotation
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            #(216,248,1,1,2,7) [z,y,x,num_] 7(z,y,x,l,w,h,theta)
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]
            
            #[z, y, x, #anchor_size, #rotation, 7]->[x,y,z,#anchor_size, #rotation, 7], 7means(x,y,z,dx,dy,dz,rot)
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            #anchors = anchors.view(-1, anchors.shape[-1])
            #move z from anchor bottom to anchor center
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        
        #all_anchors [(1,248,216,1,2,7), (1,248,216,1,2,7), (1,248,216,1,2,7)]
        #num_anchors_per_location [2,2,2]
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
