# ref: https://github.com/PRBonn/semantic-kitti-api/blob/master/visualize.py

#!/usr/bin/env python3
import argparse
import os
import yaml #pip install PyYAML
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#Current folder
ROOT_DIR = os.path.dirname(BASE_DIR)#Project root folder
sys.path.append(ROOT_DIR)

#add __init__.py in KittiSemantic
from KittiSemantic.laserscan import LaserScan, SemLaserScan
from KittiSemantic.laserscanvis import LaserScanVis
#import KittiSemantic.laserscan as LaserScan #'module' object is not callable
#import KittiSemantic.laserscanvis as LaserScanVis
# from auxiliary.laserscan import LaserScan, SemLaserScan
# from auxiliary.laserscanvis import LaserScanVis

# ls /mnt/DATA10T/Datasets/KittiSemantic/dataset/sequences/00
# calib.txt  image_2  image_3  labels  poses.txt  times.txt  velodyne

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./myvisualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default="/mnt/DATA10T/Datasets/KittiSemantic/dataset",#,"D:\Dataset\KittiSemantic\dataset"
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="./KittiSemantic/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--do_instances', '-di',
        dest='do_instances',
        default=True,#False,
        action='store_true',
        help='Visualize instances too. Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()
    
    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds, get all Lidar bin files in scan_paths
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort() #list of Lidar bin files

    label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
    if os.path.isdir(label_paths):
      print("Labels folder exists! Using labels from %s" % label_paths)
    else:
      print("Labels folder doesn't exist! Exiting...")
      quit()
    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort() #The list of all files of label names

    # check that there are same amount of labels and scans
    assert(len(label_names) == len(scan_names))

    # create a scan
    if FLAGS.ignore_semantics:
        scan = LaserScan(project=True)  # project all opened scans to spheric proj
    else:
        color_dict = CFG["color_map"]
        nclasses = len(color_dict)#34classes
        scan = SemLaserScan(nclasses, color_dict, project=True)
    
    # create a visualizer
    semantics = not FLAGS.ignore_semantics #True
    instances = FLAGS.do_instances #False
    if not semantics:
        label_names = None
    vis = LaserScanVis(scan=scan,
                        scan_names=scan_names,
                        label_names=label_names,
                        offset=FLAGS.offset,
                        semantics=semantics, instances=instances and semantics)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()