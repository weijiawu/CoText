#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2021/07/13 17:45:08
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
import os
from os import path as osp
import shutil
import logging
from zhon.hanzi import punctuation
import re, string


def strip_points(s):
    s = str(s)
    for i in string.punctuation + punctuation:
        s = s.replace(i, '')
    s = s.replace(' ', '')
    s = s.upper()
    return s

def get_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
logger = get_logger('root')


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)



def iou_polygen(g, p):
    # 取g,p中的几何体信息组成多边形
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    # 判断g,p是否为有效的多边形几何体
    if not g.is_valid or not p.is_valid:
        return 0
    # 取两个几何体的交集和并集
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union

def standard_nms(dets, thresh):
    # 标准NMS
    order = np.argsort(dets[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([JDETracker.iou_polygen(dets[i], dets[t]) for t in order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 0.1) * (y2 - y1 + 0.1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 0.1)
        h = np.maximum(0.0, yy2 - yy1 + 0.1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]