# 2D3DFusion
## System Architecture
Distributed Cooperative Sensing: Facilitates distributed deep fusion for hybrid Radar/Camera/Lidar data and multi-station cooperative sensing. The architecture of our proposed AI processing framework for 2D3DFusion is shown here:

<img width="643" alt="image" src="https://github.com/lkk688/2D3DFusion/assets/6676586/f7b69bbd-e8d5-42bf-b48f-1c964263f6ad">

## Readthedocs

The document for 3D detection model design and Kitti and Waymo data training: https://mytutorial-lkk.readthedocs.io/en/latest/mydetector3d.html

The document for 3D detection based on V2X cooperative Lidar sensing data: https://mytutorial-lkk.readthedocs.io/en/latest/3DV2X.html

Nuscence dataset and BEV transform based on Lift Splat is available on: https://mytutorial-lkk.readthedocs.io/en/latest/nuscenes.html

## Setup repo
Clone this repository, install this package (no need NVIDIA CUDA environment, tested in Mac)
```bash
git clone https://github.com/lkk688/2D3DFusion.git
python setup.py develop
```
After the package installation, you can test import the mydetector3d module
```bash
import mydetector3d
```

Install the [SharedArray library](https://pypi.org/project/SharedArray/), SparseConv library from [spconv](https://github.com/traveller59/spconv) and [numba](https://numba.pydata.org/numba-doc/latest/user/installing.html):
```bash
pip install spconv-cu117 #pip install spconv-cu118
pip install numba
```

If using HPC, you can build additional cuda ops libraries via
```bash
(mycondapy39) [010796032@cs001 3DDepth]$ module load cuda-11.8.0-gcc-11.2.0-5tlywx3 #should match pytorch cuda version
(mycondapy39) [010796032@cs001 3DDepth]$ python cudasetup.py build_ext --inplace
pip install nuscenes-devkit #required by nuscenes dataset
pip install efficientnet_pytorch==0.7.0 #required by lss
pip install pynvml
pip install nvidia-ml-py3 #required by import nvidia_smi
pip3 install --upgrade pyside2 pyqt5 #qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
pip install kornia #required by BEVFusion
$ module load gcc/11.2.0
$ python cudasetup.py build_ext --inplace
```

Install 'mayavi' (ref: https://docs.enthought.com/mayavi/mayavi/installation.html) and open3d (ref: http://www.open3d.org/docs/release/getting_started.html) for 3d point cloud visualization
```bash
conda create --name mycondapy311 python=3.11
conda activate mycondapy311
pip install pyqt6
#test pyqt6: sdrpysim/testpyqt6.py
pip install pyqtgraph
#Successfully installed numpy-1.26.1 pyqtgraph-0.13.3
#import pyqtgraph as pg
#test pyqtgraph: sdrpysim\pyqt6qtgraphtest.py
pip install matplotlib #conda install matplotlib will install pyqt5
#Successfully installed contourpy-1.2.0 cycler-0.12.1 fonttools-4.44.0 kiwisolver-1.4.5 matplotlib-3.8.1 packaging-23.2 pillow-10.1.0 pyparsing-3.1.1 python-dateutil-2.8.2 six-1.16.0
pip install opencv-python-headless
pip install mayavi
#pip3 install PySide6 #will cause pyqt6 not working, but mayavi needs PySide6
pip install pyqt5 #needed by mayavi and matplotlib
conda install -c conda-forge jupyterlab
python VisUtils/testmayavi.py #test mayavi installation
pip install open3d #does not support python3.11, only 3.7-3.10
#install development version of open3d: http://www.open3d.org/docs/latest/getting_started.html
pip install -U --trusted-host www.open3d.org -f http://www.open3d.org/docs/latest/getting_started.html open3d
# Verify installation
python -c "import open3d as o3d; print(o3d.__version__)"
# Open3D CLI
open3d example visualization/draw
python VisUtils/testopen3d.py #test open3d installation
```

[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) installation based on [installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html)
```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
(mycondapy310) PS D:\Developer> &'C:\Program Files\Git\bin\git.exe' --version
(mycondapy310) PS D:\Developer> &'C:\Program Files\Git\bin\git.exe' clone https://github.com/open-mmlab/mmdetection3d.git
(mycondapy310) PS D:\Developer\mmdetection3d> pip install -v -e .
>>> import mmdet3d
>>> print(mmdet3d.__version__)
1.4.0
pip install cumm-cu118
pip install spconv-cu118
#test installation
(mycondapy310) PS D:\Developer\mmdetection3d> mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest .
(mycondapy310) PS D:\Developer\mmdetection3d> python demo/pcd_demo.py demo/data/kitti/000008.bin pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth
cat .\outputs\preds\000008.json
{"labels_3d": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "scores_3d": 
```

## BEV Fusion Training in HPC now
```bash
(mycondapy310) [010796032@cs001 3DDepth]$ python ./mydetector3d/tools/mytrain.py
```

## Kitti Dataset
Check [kittidata](Kitti/kittidata.md) for detailed information of Kitti dataset.

## Waymo Dataset
Check [waymodata](Waymo/waymodata.md) for detailed information of Waymo dataset.

