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


class S3DIS(data.IterableDataset):
    """
    #TODO
    """
    def __init__(self, root: str, cache_dir: str,
                 split: Optional[str] = 'train'):
        assert split.lower() in ['train', 'test']
        self.root = Path(root)
        self.cache_dir = cache_dir
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

        for anno_path in self.anno_paths:
            elements = anno_path.split('/')
            out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
            out_filepath = os.path.join(cache_dir, out_filename)
            self.collect_point_label(anno_path, out_filepath)
        
        
        
    def collect_point_label(self, anno_path, out_filename):
        """ Convert original dataset files to data_label file (each line is XYZRGBL).
            We aggregated all the points from each instance in the room.
        Args:
            anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
            out_filename: path to save collected points and labels (each line is XYZRGBL)
        Returns:
            None
        Note:
            the points are shifted before save, the most negative point is now at origin.
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
        np.save(out_filename, data_label)
    
    def __len__(self):
        return len(self.anno_paths)


