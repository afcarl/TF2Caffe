# --------------------------------------------------------
# TF2Caffe, _init_paths.py
# Copyright (c) 2016
# Haozhi Qi (https://github.com/Oh233)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'models')
add_path(lib_path)
