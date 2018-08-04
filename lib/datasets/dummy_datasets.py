# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict
from helpers.labels import labels

classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.wad_classes = {1: 36, 2: 35, 3: 33, 4: 34, 6: 39, 8: 38}
    ds.confident_threshold = 0.5
    return ds


def get_wad_dataset():
    """A dummy WAD dataset that includes only the 'classes' field."""
    ds = AttrDict()
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.coco_to_this = {1: 36, 2: 35, 3: 33, 4: 34, 6: 39, 8: 38}

    return ds


def get_cityscape_dataset():
    """A dummy cityscape dataset that includes only the 'classes' field."""
    ds = AttrDict()
    ds.classes = {i: name for i, name in enumerate(classes)} 
    ds.classes_name_to_num = {name: i for i, name in enumerate(classes)} 
    ds.coco_to_this = {}
    for label in labels:
        name = label.name
        if name in classes and label.hasInstances and not label.ignoreInEval:
            ds.coco_to_this[ds.classes_name_to_num[name]] = label.id
            
    ds.confident_threshold = 0.5
    return ds