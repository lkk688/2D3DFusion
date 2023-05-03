import os
import torch
from torch.utils.data import Dataset, DataLoader
#from datautils import MyTrainDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2" #"0,1"
num_gpus= torch.cuda.device_count()
print("Device numbers:", num_gpus)
for gpuidx in range(num_gpus):
    print("Device properties:", torch.cuda.get_device_properties(gpuidx))
    print("Utilization:", torch.cuda.utilization(gpuidx))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(gpuidx)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(gpuidx)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(gpuidx)/1024**3,1), 'GB') #renamed to memory_reserved


# get index of currently selected device
print('current device:', torch.cuda.current_device()) # returns 0 in my case

DATAPATH='/data/cmpe249-fa22/torchvisiondata'
train_set = datasets.FashionMNIST(
        root=DATAPATH,
        train=True,
        download=True,
        transform=ToTensor(),
    )
batch_size = 16
train_data = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )
gpu_id=torch.device('cuda:2')
for source, targets in train_data:
    source = source.to(gpu_id)
    targets = targets.to(gpu_id)
    print(targets)
    print(f"Shape of X [N, C, H, W]: {source.shape}")
    print(f"Shape of y: {targets.shape} {targets.dtype}")
    break

print('Done')

# print("After CUDA_VISIBLE_DEVICES, Device numbers:", num_gpus)
# for gpuidx in range(num_gpus):
#     print("Device properties:", torch.cuda.get_device_properties(gpuidx))
#     print("Utilization:", torch.cuda.utilization(gpuidx))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(gpuidx)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(gpuidx)/1024**3,1), 'GB')
