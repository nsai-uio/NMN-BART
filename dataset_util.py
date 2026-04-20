import numpy as np
import json
import pickle
import torch
import math
import h5py
import re
from collections import Counter



_special_chars = re.compile('[^a-z0-9 ]*')
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

def process_punctuation(s):
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def find_vision_feat(feat_file, img_idx, feat_coco_id_to_index, use_spatial):
    index = feat_coco_id_to_index[img_idx]
    with h5py.File(feat_file, 'r') as f_file:
        vision_feat = f_file['features'][index]
        boxes = f_file['boxes'][index]
        w = f_file['widths'][index]
        h = f_file['heights'][index]
    spatial_feat = np.zeros((5, len(boxes[0])))
    spatial_feat[0, :] = boxes[0, :] * 2 / w - 1 # x1
    spatial_feat[1, :] = boxes[1, :] * 2 / h - 1 # y1
    spatial_feat[2, :] = boxes[2, :] * 2 / w - 1 # x2
    spatial_feat[3, :] = boxes[3, :] * 2 / h - 1 # y2
    spatial_feat[4, :] = (spatial_feat[2, :]-spatial_feat[0, :]) * (spatial_feat[3, :]-spatial_feat[1, :])
    if use_spatial:
        vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
    vision_feat = torch.from_numpy(vision_feat).float()
    
    num_feat = boxes.shape[1]
    relation_mask = np.zeros((num_feat, num_feat))
    for i in range(num_feat):
        for j in range(i+1, num_feat):
            # if there is no overlap between two bounding box
            if boxes[0,i]>boxes[2,j] or boxes[0,j]>boxes[2,i] or boxes[1,i]>boxes[3,j] or boxes[1,j]>boxes[3,i]:
                pass
            else:
                relation_mask[i,j] = relation_mask[j,i] = 1
    relation_mask = torch.from_numpy(relation_mask).byte()
    return vision_feat, relation_mask