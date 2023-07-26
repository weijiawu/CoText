#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   distances.py
@Time    :   2021/05/13 18:12:32
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapely.geometry import Polygon, MultiPoint
import numpy as np

def iou_matrix_polygen(objs, hyps):
    ious = np.zeros((len(objs), len(hyps)), dtype=np.float)
    if ious.size == 0:
        return ious

    objs = np.asfarray(objs)  # m
    hyps = np.asfarray(hyps)  # n
    m = objs.shape[0]
    n = hyps.shape[0]
    # 初始化一个m*n的矩阵
    iou_mat = np.zeros((m, n))
    assert objs.shape[1] == 8
    assert hyps.shape[1] == 8
    # 开始计算
    for row in range(m):
        for col in range(n):
            iou = calculate_iou_polygen(objs[row], hyps[col])
            # 更新到iou_mat
            iou_mat[row][col] = iou
    return iou_mat

def calculate_iou_polygen(bbox1, bbox2):
    '''
    :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
    :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
    :return:
    '''
    bbox1 = np.array([bbox1[0], bbox1[1],
                      bbox1[6], bbox1[7],
                      bbox1[4], bbox1[5],
                      bbox1[2], bbox1[3]]).reshape(4, 2)
    poly1 = Polygon(bbox1).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下 右下 右上 左上
    bbox2 = np.array([bbox2[0], bbox2[1],
                      bbox2[6], bbox2[7],
                      bbox2[4], bbox2[5],
                      bbox2[2], bbox2[3]]).reshape(4, 2)
    poly2 = Polygon(bbox2).convex_hull
    if poly1.area  < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = float(inter_area) / union_area
    return iou
    