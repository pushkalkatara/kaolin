# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import torch
import os
import shutil
from pathlib import Path

import kaolin as kal
from torch.utils.data import DataLoader
from kaolin.datasets import S3DIS

S3DIS_ROOT = '/scratch/pushkalkatara/data/Stanford3dDataset_v1.2_Aligned_Version/'
CACHE_DIR = '/scratch/pushkalkatara/tests/datasets/cache'

# Tests below can only be run if a ShapeNet dataset is available
S3DIS_NOT_FOUND = 'S3DIS not found at default location: {}'.format(S3DIS_ROOT)


@pytest.mark.skipif(not Path(S3DIS_ROOT).exists(), reason=S3DIS_NOT_FOUND)
def test_Points():
    points = S3DIS(
        root=S3DIS_ROOT,
        cache_dir=CACHE_DIR
    )
    print(points)
