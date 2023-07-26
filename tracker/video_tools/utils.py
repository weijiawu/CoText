#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/04/22 19:04:36
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
import os
from os import path as osp
import Levenshtein

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def adjust_rec(label_path, out_path):
    hash_table = {}
    cur_track_id = 0
    lase_content = ""
    outf = open(out_path, 'w', encoding='utf-8')
    res_list = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            if len(line) == 12:
                img_path, x1, y1, x2, y2, x3, y3, x4, y4, video_id, frame_id, track_id = line
                content = ""
            else:
                img_path, x1, y1, x2, y2, x3, y3, x4, y4, video_id, frame_id, track_id, content = line
            res_list.append([img_path, x1, y1, x2, y2, x3, y3, x4, y4, video_id, frame_id, track_id, content])
    # 排序
    res_list = sorted(res_list, key=lambda x: (x[9], x[11]))
    for line in res_list:
        img_path, x1, y1, x2, y2, x3, y3, x4, y4, video_id, frame_id, track_id, content = line
        if track_id not in hash_table:
            hash_table[track_id] = cur_track_id
            cur_track_id += 1
        else:
            # 如果是在这个，检查与上一次的内容的相似程度
            if Levenshtein.ratio(lase_content, content) < 0.3:   # 太不一样了，需要更新track_Id
                hash_table[track_id] = cur_track_id
                cur_track_id += 1
        # 写入到新的文件即可
        wline = "\t".join([img_path, x1, y1, x2, y2, x3, y3, x4, y4, video_id, frame_id, str(hash_table[track_id]), content])+"\n"
        outf.write(wline)
        lase_content = content 
    outf.close()

            


if __name__ == "__main__":
    label_path = '/Users/lzneu/workspace/video_tools/data/track_verbframe_4dre_rec1.txt'
    out_path = '/Users/lzneu/workspace/video_tools/data/track_verbframe_4dre_rec1_adjust.txt'
    adjust_rec(label_path, out_path)