#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader, modified based on https://github.com/edwardzhou130/PolarSeg/blob/master/dataloader/dataset.py 
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data

DEBUG_MODE = True

class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, configfile="semantic-kitti.yaml"):
        self.return_ref = return_ref
        with open(configfile, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        
        SemKITTI_label_name = dict()
        learningmap=semkittiyaml['learning_map']#original label number (not continuous)
        self.learning_map = learningmap
        #print(learningmap.keys())#original label ids
        for i in sorted(list(learningmap.keys()))[::-1]: #starts from the end towards the first, 259, 258
            SemKITTI_label_name[learningmap[i]] = semkittiyaml['labels'][i] #label name
        self.SemKITTI_label_name = SemKITTI_label_name #dict new id to name 5: 'moving-other-vehicle'

        self.imageset = imageset #sequence list in yaml file
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort() #all lidar file list
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4)) #read Lidar file, 4: xyzintensity
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1)) #load labels, size=(points,1) int32
            annotated_data = annotated_data & 0xFFFF #delete high 16 digits binary, only get label id, not instance id
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data) #map to new label
        data_tuple = (raw_data[:,:3], annotated_data.astype(np.uint8)) #size=2 array, lidarxyz, new label
        if self.return_ref:
            data_tuple += (raw_data[:,3],) #add Lidar intensity data, size=3
        return data_tuple

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

class voxel_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
            fixed_volume_space= False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset #SemKITTI instance
        self.grid_size = np.asarray(grid_size) #python list to numpy array
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        max_bound = np.percentile(xyz,100,axis = 0) #Compute the q-th percentile of the data along the specified axis.
        min_bound = np.percentile(xyz,0,axis = 0)
        
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound #100, 100, 4.5, KITTI Lidar coordinate, x to the front, y to the left, z is up
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1) #[0.20876827, 0.27855153, 0.14516129], each grid distance
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) #(Points,3) shift the original, convert from meter to grid step (int64)
        if DEBUG_MODE==True:
            print(grid_ind[1:20,:]) #each element is one array (xyz), int, each point map to each voxel, one voxel multiple points
        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32) # voxel position same to grid size (480, 360, 32)
        dim_array = np.ones(len(self.grid_size)+1,int) #[1, 1, 1, 1]
        dim_array[0] = -1 #[-1,  1,  1,  1]
        if DEBUG_MODE:
            tmp=intervals.reshape(dim_array)
            print(tmp.shape) #(3, 1, 1, 1)
            tmp2=np.indices(self.grid_size) # grid_size=[480, 360, 32]
            print(tmp2.shape) #(3, 480, 360, 32)
        gridinterval=np.indices(self.grid_size)*intervals.reshape(dim_array)
        voxel_position = gridinterval + min_bound.reshape(dim_array) #Return an array representing the indices of a grid, (3, 480, 360, 32)
        #shape(3, 480, 360, 32), map to the original distance in meter
        if DEBUG_MODE==True:
            print(voxel_position.shape)
            print(voxel_position[0, 1:20,1:20,:])
            print(voxel_position[1, 1:20,1:20,:])
            print(voxel_position[2, 1:20,1:20,:])
        


        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label #ignored label is 255
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1) # (points,4) add label
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:] #sort
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair) #(480, 360, 32), very sparse, counts of points in each location
        if DEBUG_MODE==True:
            print(processed_label.shape)
            print(processed_label[1:20,1:20,:])
        
        data_tuple = (voxel_position,processed_label) #array size=2, voxel position and label

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound #(Points,3)
        return_xyz = xyz - voxel_centers
        if DEBUG_MODE==True:
            print(return_xyz[1:20,:])
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1) #(Points,6)
        
        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            signew=sig[...,np.newaxis] #(points,1)
            return_fea = np.concatenate((return_xyz,signew),axis = 1) #add intensity #(Points,7)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple 
        #array 5: voxel position (3, 480, 360, 32), 
        #         process_label of voxel (480, 360, 32), counts of points
        #         grid_ind points grid step (Points,3), 
        #         original label (Points,1)
        #         Points feature (Points,7) return_xyz (center data on each voxel),xyz,itensity

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

class spherical_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
               fixed_volume_space= False, max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)
        
        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)
        
        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label[grid_ind[:,0],grid_ind[:,1],grid_ind[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[grid_ind[:,2],grid_ind[:,0],grid_ind[:,1]]=1.

        data_tuple = (distance_feature,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple
    
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False) #optimization by Numba’s JIT compiler
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16) #(256,0)
    if DEBUG_MODE:
        print(sorted_label_voxel_pair[0,3]) #(points,4)
    counter[sorted_label_voxel_pair[0,3]] = 1 #first position is 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3] #[0, 0, 0]
    for i in range(1,sorted_label_voxel_pair.shape[0]): # for all points, each point is represend in voxel index
        cur_ind = sorted_label_voxel_pair[i,:3] #voxel index
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1 # label position +1
    if DEBUG_MODE:
        print(cur_sear_ind) #[398, 359, 30]
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index


if __name__ == '__main__':
    #test the dataset
    # load Semantic KITTI class info
    configfile="KittiSemantic/semantic-kitti.yaml"
    with open(configfile, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    learningmap=semkittiyaml['learning_map']
    print(learningmap.keys())#original label ids
    for i in sorted(list(learningmap.keys()))[::-1]: #starts from the end towards the first, 259, 258
        SemKITTI_label_name[learningmap[i]] = semkittiyaml['labels'][i] #label name, dict new id to name 5: 'moving-other-vehicle'
    
    #Test SemKITTI
    data_path = "/mnt/DATA10T/Datasets/KittiSemantic/dataset/" #datapath in i9 machine
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, configfile=configfile)
    print(val_pt_dataset.SemKITTI_label_name)
    lidarfile_list=val_pt_dataset.im_idx
    data=val_pt_dataset[0] #SemKITTI getitem function to read Lidar file
    if len(data) == 2:
        xyz,labels = data
    elif len(data) == 3:
        xyz,labels,sig = data
        print(sig.shape)
        if len(sig.shape) == 2: sig = np.squeeze(sig)
        print(sig.shape)
    else: raise Exception('Return invalid data tuple')

    #Test voxel_dataset
    grid_size = [480,360,32]
    val_dataset=voxel_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
    voxeldata=val_dataset[0] #voxel_dataset getitem function

    print('End')
