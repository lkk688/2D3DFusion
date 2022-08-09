from glob import glob
import time
import os
from pathlib import Path
import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    raise ImportError("matplotlib error")
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

Sensor_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT','SIDE_LEFT', 'SIDE_RIGHT']

objecttype_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
INSTANCE_Color = {
    'UNKNOWN':'black', 'VEHICLE':'red', 'PEDESTRIAN':'green', 'SIGN': 'yellow', 'CYCLIST':'purple'
}#'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'

def show_camera_image(camera_image, camera_labels, camera_name_str, layout, cmap=None):
    """Show a camera image and the given camera labels."""
    ax = plt.subplot(*layout)
   
    # Iterate over the individual labels.
    for label in camera_labels:
        #print(label.type) #1 for vehicle
        label_type=int(label[0])
        xmin, ymin, xmax, ymax=label[1:]
        objectclass=objecttype_list[label_type]
        colorlabel=INSTANCE_Color[objectclass]
        width = xmax-xmin #x-axis
        height = ymax-ymin #y-axis
        # Draw the object bounding box.
        ax.add_patch(patches.Rectangle(
            xy=(xmin,ymin),
            width=width,
            height=height,
            linewidth=1,
            edgecolor=colorlabel,
            facecolor='none'))
        ax.text(int((xmin+xmax)/2), ymin, objectclass, color=colorlabel, fontsize=8)

    # Show the camera image.
    plt.imshow(camera_image, cmap=cmap)
    plt.title(camera_name_str)
    plt.grid(False)
    plt.axis('off')

def parseSemanticDictfile(base_dir, filename):
    Final_array=np.load(base_dir / filename, allow_pickle=True, mmap_mode='r')
    data_array=Final_array['arr_0']
    array_len=len(data_array)
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))

    #for frameid in range(array_len):
    frameid=0
    print("frameid:", frameid)
    convertedframesdict = data_array[frameid] #{'key':key, 'context_name':context_name, 'framedict':framedict}
    context_name=convertedframesdict['context_name']
    print("Context name:", context_name)
    for key, value in convertedframesdict.items():
        print(key)
        print(type(value))
    
    #['points']
    points_all = convertedframesdict['points_all']
    point_labels_all = convertedframesdict['point_labels_all']
    cp_points_all = convertedframesdict['cp_points_all']# camera projection corresponding to each point.
    
    #images
    getimages_fromdict(convertedframesdict)

# For each camera:
    #     <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
    #     <CAMERA_NAME>_INTRINSIC: 9 float32 array
    #     <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
    #     <CAMERA_NAME>_WIDTH: int64 scalar
    #     <CAMERA_NAME>_HEIGHT: int64 scalar
    #     <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
    #     <CAMERA_NAME>_POSE: 4x4 float32 array
    #     <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
def getimages_fromdict(framedict):
    plt.figure(figsize=(25, 20))
    #FRONT_LEFT_IMAGE, SIDE_LEFT_IMAGE, FRONT_RIGHT_IMAGE, SIDE_RIGHT_IMAGE
    for index, cam_name_str in enumerate(Sensor_list):
        camera_labels=framedict[f'{cam_name_str}_camera_label'] #label.type, xmin, ymin, xmax, ymax
        camera_image=framedict[f'{cam_name_str}_IMAGE']
        camera_pose=framedict[f'{cam_name_str}_POSE']
        camera_intrinsic=framedict[f'{cam_name_str}_INTRINSIC']
        camera_intrinsic=framedict[f'{cam_name_str}_EXTRINSIC']
        camera_width=framedict[f'{cam_name_str}_WIDTH']
        camera_height=framedict[f'{cam_name_str}_HEIGHT']

        show_camera_image(camera_image, camera_labels, cam_name_str, [3, 3, index+1])
        

def saveimagetofile(image, frame_idx, file_idx, basepath, foldername):
    fullfolderpath=os.path.join(basepath,foldername)
    img_path = fullfolderpath + \
        f'{str(file_idx).zfill(3)}' + \
        f'{str(frame_idx).zfill(3)}.png'
    if not os.path.exists(fullfolderpath):
        os.makedirs(fullfolderpath)
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    base_dir="/data/cmpe249-fa22/Waymo132KittiSematic/" 
    base_dir = Path(base_dir)
    filename="train0_10017090168044687777_6380_000_6400_000.npz"

    parseSemanticDictfile(base_dir, filename)
    