# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

import torch.utils.data as data
from imageio import imread
from path import Path
import random
import cv2
import time
import torch
import pdb
import st_utils.frame_utils as frame_utils

HEIGHT, WIDTH = 192,640 # 320,896
TAG_FLOAT = 202021.25

def load_as_flow(src_file):
    #assert(os.path.exists(src_file))

    if src_file.lower().endswith('.flo'):

        with open(src_file, 'rb') as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert(tag == TAG_FLOAT)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith('.png'):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = (flow_raw[:, :, 0] == 0)
        flow[invalid, :] = 0

    else:
        raise IOError
    return flow


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        #print('getting image path')
        #print(self.get_image_path(folder, frame_index, side))
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

def sample_pairs_with_flow(h, w, validity_mask, num_sample_points=100000, d_lim=30, flow_lim=100, norm_lim=3):
    #validity_mask_valid= np.argwhere(validity_mask==1) 

    validity_mask_valid = (torch.squeeze(validity_mask)).nonzero()
    idx = torch.randint(low=0, high=validity_mask_valid.shape[0], size=(num_sample_points, 2))
    idx = idx.type(torch.LongTensor)

    first = validity_mask_valid[idx[:,0],:]
    second = validity_mask_valid[idx[:,1],:]
    diff = (first-second).type(torch.FloatTensor)
    dist = torch.norm(diff, p=2, dim=1)
    selected_points = idx[dist<d_lim]
    #selected_points = idx
    p1 = validity_mask_valid[selected_points[:,0],:].type(torch.LongTensor)[:100000,:]
    p2 = validity_mask_valid[selected_points[:,1],:].type(torch.LongTensor)[:100000,:]
    return p1, p2
    

'''    
def sample_pairs_with_flow(h, w, validity_mask, num_sample_points=100000):
    #pdb.set_trace()
    
    validity_mask_valid = (torch.squeeze(validity_mask)).nonzero()
    idx = torch.randint(low=0, high=validity_mask_valid.shape[0], size=(num_sample_points, 2))
    idx = idx.type(torch.LongTensor)
    p1 = validity_mask_valid[idx[:,0],:].type(torch.LongTensor)
    p2 = validity_mask_valid[idx[:,1],:].type(torch.LongTensor)
    return p1, p2
'''

def two_frames_checker(flowAB):
    h,w = flowAB.shape[0:2]
    x_mat = (np.expand_dims(range(w),0) * np.ones((h,1),dtype=np.int32)).astype(np.int32)
    y_mat = (np.ones((1,w),dtype=np.int32) * np.expand_dims(range(h),1)).astype(np.int32)

    d1 = flowAB

    r_cords = (y_mat + d1[:,:,1]).astype(np.int32)
    c_cords = (x_mat + d1[:,:,0]).astype(np.int32)
    mask1 = r_cords>h-1
    mask2 = r_cords<0
    mask3 = c_cords>w-1
    mask4 = c_cords<0

    r_cords[mask1] = h-1
    r_cords[mask2] = 0
    c_cords[mask3] = w-1
    c_cords[mask4] = 0

    corresp = np.dstack((r_cords,c_cords))
    valid_map = (1-mask1) * (1-mask2) * (1-mask3) * (1-mask4) 
    return corresp, valid_map


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        #print('entered here')
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_flow_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, '.flo')
        image_path = os.path.join(
            '/cluster/scratch/takmaza/CVL/RAFT_Flow_mini/vec', folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path  

    def get_flow(self, folder, frame_index, side):
        fl_path = self.get_flow_path(folder, frame_index, side)
        #[frame_utils.read_gen(self.flow_list[index][img_ind]) for img_ind in list(range(self.num_frames-1))]

        #print('getting flow path')
        #print(fl_path)
        loaded_flow = np.asarray(load_as_flow(fl_path))
        #pdb.set_trace()

        #print('loaded the flow')
        #print('flow size: ', loaded_flow.shape)
        corresp, valid_map = two_frames_checker(loaded_flow)
        corresp = torch.from_numpy(corresp.astype(np.int32))#.to(self.device)
        valid_map = torch.from_numpy(valid_map)#.to(self.device)
        loaded_flow = torch.from_numpy(loaded_flow.astype(np.float32))#.to(self.device)

        p1, p2 = sample_pairs_with_flow(h=self.height, w=self.width, validity_mask=valid_map, num_sample_points=self.args.num_pairs, d_lim=self.args.d_lim)
        return loaded_flow, corresp, valid_map, p1, p2


    def check_if_flow_exists(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, '.flo')
        image_path = os.path.join(
            '/cluster/scratch/takmaza/CVL/RAFT_Flow_mini/vec', folder, "image_0{}/data".format(self.side_map[side]), f_str)
        if os.path.exists(image_path):
            return True
        else:
            return False



class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
