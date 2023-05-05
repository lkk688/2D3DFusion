import os
import torch
from torch.utils.data import Dataset, DataLoader
#from datautils import MyTrainDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import nvidia_smi #pip install nvidia-ml-py3

nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(GB total), {} (GB free), {} (GB used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total/1024**3, info.free/1024**3, info.used/1024**3))

nvidia_smi.nvmlShutdown()

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
    #print('Cached:   ', round(torch.cuda.memory_cached(gpuidx)/1024**3,1), 'GB') #renamed to memory_reserved


# get index of currently selected device
print('current device:', torch.cuda.current_device()) # returns 0 in my case

gpuid=1
torch.cuda.set_device(gpuid) #model.cuda(args.gpuid)

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
#device=torch.device('cuda:0')
device = torch.device('cuda:{}'.format(gpuid))
for source, targets in train_data:
    source = source.to(device)
    targets = targets.to(device)
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
