from glob import glob
import time
import os
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    base_dir="/DATA5T2/Datasets/Waymo132/Outdicts/"#"/mnt/DATA10T/Datasets/Waymo132/Outdicts"
    base_dir = Path(base_dir)
    filename="train01234_12844373518178303651_2140_000_2160_000.npz"#"training0000__10017090168044687777_6380_000_6400_000.npy.npz"

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
    for key, value in convertedframesdict.items():
        print(key)
        print(type(value))