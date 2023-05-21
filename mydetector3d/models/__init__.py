from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia #https://github.com/kornia/kornia  pip install kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

#https://kornia.readthedocs.io/en/latest/_modules/kornia/utils/image.html#image_to_tensor
def image_to_tensor(image, keepdim: bool = True):
    """Convert a numpy image to a PyTorch 4d tensor image.

    Args:
        image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim: If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`.

    Returns:
        tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.

    Example:
        >>> img = np.ones((3, 3))
        >>> image_to_tensor(img).shape
        torch.Size([1, 3, 3])

        >>> img = np.ones((4, 4, 1))
        >>> image_to_tensor(img).shape
        torch.Size([1, 4, 4])

        >>> img = np.ones((4, 4, 3))
        >>> image_to_tensor(img, keepdim=False).shape
        torch.Size([1, 3, 4, 4])
    """
    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(f"Cannot process image with shape {input_shape}")

    return tensor.unsqueeze(0) if not keepdim else tensor

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            #batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            #batch_dict[key] = image_to_tensor(val).float().cuda().contiguous()
            batch_dict[key] = image_to_tensor(val).float().cuda()
            print(type(batch_dict[key]))
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict']) #define a 'ModelReturn' tuple and contains properties of (loss, tb_dict, disp_dict)

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func