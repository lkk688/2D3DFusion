import os.path as osp
from functools import cmp_to_key
import logging

from base_dataset import DAIRV2XDataset, get_annos, build_path_to_info
from mydetector3d.utils.dataset_utils import load_json
from frame import InfFrame, VehFrame, VICFrame, Label

logger = logging.getLogger(__name__)

class DAIRV2XI(DAIRV2XDataset):
    def __init__(self, path, args, split="train", sensortype="lidar", extended_range=None):
        super().__init__(path, args, split, extended_range)
        data_infos = load_json(osp.join(path, "infrastructure-side/data_info.json"))
        split_path = args.split_data_path
        data_infos = self.get_split(split_path, split, data_infos)

        self.inf_path2info = build_path_to_info(
            "",
            load_json(osp.join(path, "infrastructure-side/data_info.json")),
            sensortype,
        )

        self.data = []
        for elem in data_infos:
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
