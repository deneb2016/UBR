# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for OICR
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import lib.voc.pascal_voc
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                lib.voc.pascal_voc(split, year))

def get_imdb(name):
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
