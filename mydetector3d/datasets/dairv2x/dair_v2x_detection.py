import os.path as osp
from functools import cmp_to_key
import logging

from base_dataset import DAIRV2XDataset, get_annos, build_path_to_info, Label
from mydetector3d.utils.dataset_utils import load_json
from frame import InfFrame, VehFrame, VICFrame
from v2x_utils import RectFilter, Filter

logger = logging.getLogger(__name__)

class DAIRV2XI(DAIRV2XDataset):
    def __init__(self, path, args, split="train", sensortype="lidar", extended_range=None):
        super().__init__(path, args, split, extended_range)
        data_infos = load_json(osp.join(path, "infrastructure-side/data_info.json"))
        split_path = args.split_data_path #data/split_datas/example-cooperative-split-data.json'
        data_infos = self.get_split(split_path, split, data_infos)

        self.inf_path2info = build_path_to_info(
            "",
            load_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )

        self.data = []
        for elem in data_infos: #selected 15 infos
            gt_label = {}
            filt = RectFilter(extended_range[0]) if extended_range is not None else Filter()
            gt_label["camera"] = Label(osp.join(path, "infrastructure-side", elem["label_camera_std_path"]), filt)
            gt_label["lidar"] = Label(osp.join(path, "infrastructure-side", elem["label_lidar_std_path"]), filt)

            self.data.append((InfFrame(path, elem), gt_label, filt))

            if sensortype == "camera":
                inf_frame = self.inf_path2info[elem["image_path"]]
                get_annos(path + "/infrastructure-side", "", inf_frame, "camera")

    def get_split(self, split_path, split, data_infos):
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data[split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for data_info in data_infos:
            frame_idx = data_info["image_path"].split("/")[-1].replace(".jpg", "")
            if frame_idx in split_data:
                frame_pairs_split.append(data_info)

        return frame_pairs_split

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class DAIRV2XV(DAIRV2XDataset):
    def __init__(self, path, args, split="train", sensortype="lidar", extended_range=None):
        super().__init__(path, args, split, extended_range)
        data_infos = load_json(osp.join(path, "vehicle-side/data_info.json"))
        split_path = args.split_data_path
        data_infos = self.get_split(split_path, split, data_infos)

        self.veh_path2info = build_path_to_info(
            "",
            load_json(osp.join(path, "vehicle-side/data_info.json")),
            sensortype,
        )

        self.data = []
        for elem in data_infos:
            gt_label = {}
            filt = RectFilter(extended_range[0]) if extended_range is not None else Filter
            for view in ["camera", "lidar"]:
                gt_label[view] = Label(osp.join(path, "vehicle-side", elem["label_" + view + "_std_path"]), filt)

            self.data.append((VehFrame(path, elem), gt_label, filt))

            if sensortype == "camera":
                veh_frame = self.veh_path2info[elem["image_path"]]
                get_annos(path + "/vehicle-side", "", veh_frame, "camera")

    def get_split(self, split_path, split, data_infos):
        if osp.exists(split_path):
            split_data = load_json(split_path)
        else:
            print("Split File Doesn't Exists!")
            raise Exception

        if split in ["train", "val", "test"]:
            split_data = split_data[split]
        else:
            print("Split Method Doesn't Exists!")
            raise Exception

        frame_pairs_split = []
        for data_info in data_infos:
            frame_idx = data_info["image_path"].split("/")[-1].replace(".jpg", "")
            if frame_idx in split_data:
                frame_pairs_split.append(data_info)

        return frame_pairs_split

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def add_arguments(parser):
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--split-data-path", type=str, default="/data/cmpe249-fa22/DAIR-C/split_datas/example-single-infrastructure-split-data.json" #"../data/split_datas/example-cooperative-split-data.json"
    )
    parser.add_argument("--dataset", type=str, default="vic-sync")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--pred-classes", nargs="+", default=["car"])
    parser.add_argument("--model", type=str, default="single_veh")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-point-cloud", action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--extended-range", type=float, nargs="+", default=[-10, -49.68, -3, 79.12, 49.68, 1])
    parser.add_argument("--sensortype", type=str, default="lidar")
    parser.add_argument("--eval-single", action="store_true")
    parser.add_argument("--val-data-path", type=str, default="", help="Help evaluate feature flow net")
    parser.add_argument("--test-mode",  type=str, default="FlowPred", help="Feature Flow Net mode: {'FlowPred', 'OriginFeat', 'Async'}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()

    from tqdm import tqdm
    import numpy as np

    input = "/data/cmpe249-fa22/DAIR-C/cooperative-vehicle-infrastructure/" #"../data/cooperative-vehicle-infrastructure/"
    split = "val"
    sensortype = "camera"
    box_range = np.array([-10, -49.68, -3, 79.12, 49.68, 1])
    indexs = [
        [0, 1, 2],
        [3, 1, 2],
        [3, 4, 2],
        [0, 4, 2],
        [0, 1, 5],
        [3, 1, 5],
        [3, 4, 5],
        [0, 4, 5],
    ]
    extended_range = np.array([[box_range[index] for index in indexs]]) #(1,8,3)
    dataset = DAIRV2XI(input, args, split, sensortype, extended_range=extended_range)

    for Frame_data, label, filt in tqdm(dataset):
        veh_image_path = Frame_data.vehicle_frame()["image_path"][-10:-4]
        inf_image_path = Frame_data.infrastructure_frame()["image_path"][-10:-4]
        print(veh_image_path, inf_image_path)