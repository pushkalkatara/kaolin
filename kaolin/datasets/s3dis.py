# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Optional

import torch
import torch.utils.data as data
import os
from pathlib import Path
from glob import glob

import numpy as np

from kaolin.rep.PointCloud import PointCloud
from .base import KaolinDataset


class S3DIS(data.Dataset):
    """
    #TODO
    """
    def __init__(self, root: str,
                 split: Optional[str] = 'train', 
                 num_points: int = 4096
    ):
        assert split.lower() in ['train', 'test']
        self.root = Path(root)
        self.num_points = num_points
        self.class_color = {
            'ceiling':  [0,255,0],
            'floor':    [0,0,255],
            'wall':     [0,255,255],
            'beam':     [255,255,0],
            'column':   [255,0,255],
            'window':   [100,100,255],
            'door':     [200,200,100],
            'table':    [170,120,200],
            'chair':    [255,0,0],
            'sofa':     [200,100,100],
            'bookcase': [10,200,100],
            'board':    [200,200,200],
            'clutter':  [50,50,50]
        }
        self.classes = list(self.class_color.keys())
        self.class2label = {cls: i for i, cls in enumerate(self.classes)}
        self.anno_paths = glob(str(self.root) + "/*/*/Annotations/")
    
    def room2block(self, data_label, num_points, block_size, stride):
        data = data_label[:, 0:6]
        label = data_label[:,-1].astype(np.uint8)
        assert(stride<=block_size)
        limit = np.amax(data, 0)
        limit = limit[0:3]
        # Get the corner location for our sampling blocks
        xbeg_list = []
        ybeg_list = []
        # implement random sample here
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
        
        # collect blocks
        block_data_list = []
        block_label_list = []
        idx = 0

        for idx in range(len(xbeg_list)):
            xbeg = xbeg_list[idx]
            ybeg = ybeg_list[idx]
            xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
            ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
            cond = xcond & ycond
            if np.sum(cond) < 100:
                continue
            
            block_data = data[cond, :]
            block_label = label[cond]

            #randomly subsample data
            block_data_sampled, block_label_sampled = self.sample_data_label(
                block_data, block_label, num_points
            )
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
        
        return np.concatenate(block_data_list, 0), \
               np.concatenate(block_label_list, 0)
        
    def sample_data_label(self, data, label, num_sample):
        new_data, sample_indices = self.sample_data(data, num_sample)
        new_label = label[sample_indices]
        return new_data, new_label
    
    def sample_data(self, data, num_sample):
        """ data is in N x ...
            we want to keep num_samplexC of them.
            if N > num_sample, we will randomly keep num_sample of them.
            if N < num_sample, we will randomly duplicate samples.
        """
        N = data.shape[0]
        if (N == num_sample):
            return data, range(N)
        elif (N > num_sample):
            sample = np.random.choice(N, num_sample)
            return data[sample, ...], sample
        else:
            sample = np.random.choice(N, num_sample-N)
            dup_data = data[sample, ...]
            return np.concatenate([data, dup_data], 0), range(N)+list(sample)

    def collect_point_label(self, anno_path):
        """ Convert original dataset files to data_label file (each line is XYZRGBL).
            We aggregated all the points from each instance in the room.
        Args:
            anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
            out_filename: path to save collected points and labels (each line is XYZRGBL)
        Returns:
            XYZRGBL
        Note:
            the points are shifted, the most negative point is now at origin.
        """
        points_list = []
        for f in glob(os.path.join(anno_path, '*.txt')):
            cls = os.path.basename(f).split('_')[0]
            if cls not in self.classes: # note: in some room there is 'staris' class.
                cls = 'clutter'
            points = np.loadtxt(f)
            labels = np.ones((points.shape[0],1)) * self.class2label[cls]
            points_list.append(np.concatenate([points, labels], 1))
        data_label = np.concatenate(points_list, 0)
        xyz_min = np.amin(data_label, axis=0)[0:3]
        data_label[:, 0:3] -= xyz_min
        return data_label
    
    def __len__(self):
        return len(self.anno_paths)
    
    def __getitem__(self, room_idx):
        """
        room idx: ID of room
        """
        anno_path = self.anno_paths[room_idx]
        print(anno_path)
        data_label = self.collect_point_label(anno_path)
        data, label = self.room2block(data_label, self.num_points, \
                        block_size=1.0, stride=0.5)
        print(data, label)
        return data, label