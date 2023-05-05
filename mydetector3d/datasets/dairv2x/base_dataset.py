import os.path as osp
import json
import os

from torch.utils.data import Dataset
#from v2x_utils import get_trans
#from dataset.dataset_utils import load_json
from mydetector3d.utils.dataset_utils import load_json

def get_trans(info):
    return info["translation"], info["rotation"]

def get_annos(path, prefix, single_frame, sensortype="camera"):
    img_path = path + prefix + single_frame["image_path"]
    trans0_path = ""
    if "calib_lidar_to_camera_path" in single_frame.keys():
        trans0_path = single_frame["calib_lidar_to_camera_path"]
    else:
        trans0_path = single_frame["calib_virtuallidar_to_camera_path"]
    trans1_path = single_frame["calib_camera_intrinsic_path"]
    trans0, rot0 = get_trans(load_json(osp.join(path, prefix, trans0_path)))
    lidar2camera = {}
    lidar2camera.update(
        {
            "translation": trans0,
            "rotation": rot0,
        }
    )
    # trans0, rot0 = lidar2camera["translation"], lidar2camera["rotation"]
    camera2image = load_json(osp.join(path, prefix, trans1_path))["cam_K"]

    annFile = {}
    img_ann = {}
    calib = {}
    calib.update(
        {
            "cam_intrinsic": camera2image,
            "Tr_velo_to_cam": lidar2camera,
        }
    )

    img_ann.update({"file_name": img_path, "calib": calib})
    imglist = []
    imglist.append(img_ann)
    annFile.update({"images": imglist})
    if not osp.exists(osp.join(path, prefix, "annos")):
        os.mkdir(osp.join(path, prefix, "annos"))
    ann_path_o = osp.join(path, prefix, "annos", single_frame["image_path"].split("/")[-1].split(".")[0] + ".json")
    with open(ann_path_o, "w") as f:
        json.dump(annFile, f)


def build_path_to_info(prefix, data, sensortype="lidar"):
    path2info = {}
    if sensortype == "lidar":
        for elem in data:
            if elem["pointcloud_path"] == "":
                continue
            path = osp.join(prefix, elem["pointcloud_path"])
            path2info[path] = elem
    elif sensortype == "camera":
        for elem in data:
            if elem["image_path"] == "":
                continue
            path = osp.join(prefix, elem["image_path"])
            path2info[path] = elem
    return path2info


class DAIRV2XDataset(Dataset):
    def __init__(self, path, args, split="train", extended_range=None):
        super().__init__()

        self.split = None

#from https://github.com/AIR-THU/DAIR-V2X/blob/main/v2x/dataset/dataset_utils/label.py
from v2x_utils import get_3d_8points
import numpy as np

#from DAIR-V2X/v2x/config.py
name2id = {
    "car": 2,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "cyclist": 1,
    "tricyclist": 3,
    "motorcyclist": 3,
    "barrow": 3,
    "barrowlist": 3,
    "pedestrian": 0,
    "trafficcone": 3,
    "pedestrianignore": 3,
    "carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}

superclass = {
    -1: "ignore",
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "ignore",
}

class Label(dict):
    def __init__(self, path, filt):
        raw_labels = load_json(path)
        boxes = []
        class_types = []
        for label in raw_labels:
            size = label["3d_dimensions"]
            if size["l"] == 0 or size["w"] == 0 or size["h"] == 0:
                continue
            if "world_8_points" in label:
                box = label["world_8_points"]
            else:
                pos = label["3d_location"]
                box = get_3d_8points(
                    [float(size["l"]), float(size["w"]), float(size["h"])],
                    float(label["rotation"]),
                    [float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(size["h"]) / 2],
                ).tolist()
            # determine if box is in extended range
            if filt is None or filt(box):
                boxes.append(box)
                class_types.append(name2id[label["type"].lower()])
        boxes = np.array(boxes)
        class_types = np.array(class_types)
        # if len(class_types) == 1:
        #     boxes = boxes[np.newaxis, :]
        self.__setitem__("boxes_3d", boxes)
        self.__setitem__("labels_3d", class_types)
        self.__setitem__("scores_3d", np.ones_like(class_types))