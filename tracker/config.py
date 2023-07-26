#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/07/25 10:58:20
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''
# ICDAR15
config = {
        # tracker相关参数
        "tracker": {
            "perform_reid": True,
            "perform_iou":True,
            "perform_kalman": False,
            "max_time_lost": 8, # 10
            "embedding_distance_thresh" : 0.5,    # small is more similar  0.5
            "IOU_distance_thresh": 0.4,           # small is more similar  0.4
            "IOU_similarity_thresh": 0.4,   # 0.4
            "min_char_tolerance": 3   # 3
        },
        # 跟踪后识别相关参数
#         "min_track_text_score": 0.85,
        # IO相关参数
        "frame_rate": 5
    }

# BOVText
# config = {
#         # tracker相关参数
#         "tracker": {
#             "perform_reid": True,
#             "perform_iou":True,
#             "perform_kalman": False,
#             "max_time_lost": 20, # 10
#             "embedding_distance_thresh" : 0.6,    # small is more similar  0.5
#             "IOU_distance_thresh": 0.5,           # small is more similar  0.4
#             "IOU_similarity_thresh": 0.4,   # 0.4
#             "min_char_tolerance": 3   # 3
#         },
#         # 跟踪后识别相关参数
# #         "min_track_text_score": 0.85,
#         # IO相关参数
#         "frame_rate": 5
#     }