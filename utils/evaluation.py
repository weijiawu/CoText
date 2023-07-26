#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2021/06/01 20:07:54
@Author  :   lzneu
@Co-Au   :   tangyejun
@Version :   3.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   视频OCR评价指标体系

概念定义：
	- 视频文本对象：视频中存在的时间连续、空间连续、内容连续的文本对象。每个文本对象由一个包含了时间、空间、内容的轨迹表示，形如: [ (frame_1, box_1, text_1), (frame_2, box_2, text_2), ... (frame_i, box_i, text_i) ]
	- 命中：fds（时间IoU ≥ 0.5、空间IoU ≥ 0.5，内容匹配度 ≥ 0.9，内容匹配度的计算方式是:  1 - （编辑距离) / (较长文本的长度) ，最小取0）
整体指标：
    - 准确率  = （命中的文本对象数量） / （预测输出的文本对象数量）
    - 召回率	召回率 =（命中的文本对象数量）/ （ 真值文本对象总量）	 
'''

# here put the import lib
import os
from os import WIFCONTINUED, path as osp
import numpy as np
import copy
from numpy.lib.function_base import append
from tqdm import  tqdm
from shapely.geometry import Polygon, MultiPoint
import Levenshtein
import json
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re, string
from zhon.hanzi import punctuation


class Evaluator(object):

    def __init__(self, gt_path, pre_path, 
                        data_type='videotext', 
                        det_only=False,
                        seqs=[], 
                        piou_thresh=0.5, 
                        sim_thresh=0.9, 
                        tiou_thresh=0.9):
        """
        @description  :
        ---------
        @param  :
            - piou_thresh: 空间iou阈值，默认为0.5
            - sim_thresh: 内容匹配度阈值，默认是0.9
            - data_type: 目前只做文本
        -------
        @Returns  : 
        -------
        """
        
        self.gt_dict, self.num_gt_objs = self.load_anno(gt_path)
        self.video_id_list = list(seqs)
        self.pre_dict, self.num_pre_objs = self.load_anno(pre_path, valid_ids=self.video_id_list)
        self.data_type = data_type
        self.eval_content = not det_only 
        self.piou_thresh = piou_thresh
        self.tiou_thresh = tiou_thresh
        self.sim_thresh = sim_thresh
        self.metric_names = ["HIT", "PRE_total", "Precision", "GT_total", "Recall", "F1"]
        self.register_names = ["TP", "FP", "FN", "IGNORE"]
        self.leader_board = {}
        self.register = {}
        self.init_metrics()
        # self.eval()


    def init_metrics(self):
        for video_id in self.video_id_list:
            self.leader_board[video_id] = {}
            self.register[video_id] = {}
            for metric_name in self.metric_names:
                self.leader_board[video_id][metric_name] = 0
            for register_name in self.register_names:
                self.register[video_id][register_name] = []
 
    def load_anno(self, file_path, start_id=0, valid_ids=None):
        
        res_dict = {}
        with open(file_path, 'r', encoding='utf-8') as j:
            video_dict = json.load(j)
        # 检查和重建track_id
        cur_track_id = 0
        for video_id, track_dict in video_dict.items():
            if valid_ids is not None and video_id not in valid_ids:
                continue
            res_dict[video_id] = {}
            for track_id, track_object in track_dict.items():
                res_dict[video_id][cur_track_id] = track_object
                cur_track_id += 1

        return res_dict, cur_track_id
    
    def eval(self):
        # 逐video计算评价指标
        for video_id in tqdm(self.video_id_list):
            gt_track_objs = list(self.gt_dict[video_id].values())
            pre_track_objs = list(self.pre_dict[video_id].values())
            
            ignore_gt_track_objs = []
            ignore_pre_track_objs = []
            valid_gt_track_objs = []
            valid_pre_track_objs = []

            for index in range(len(gt_track_objs)):
                gt_track_obj = gt_track_objs[index]
                content = gt_track_obj['text']
                tracks = gt_track_obj['tracks']
                if content == "#null":
                    ignore_gt_track_objs.append(gt_track_obj)
                else:
                    valid_gt_track_objs.append(gt_track_obj)
            
            # 优先匹配有效的gt_track_obj
            matched_gt_objs, matched_pre_objs, missed_gt_objs, missed_pre_objs = self.eval_video(valid_gt_track_objs, pre_track_objs)
            # 未匹配到的预测对象，再与ignore进行计算匹配
            _, ignore_pre_track_objs, _, missed_pre_objs = self.eval_video(ignore_gt_track_objs, missed_pre_objs, ignore_mode=True)
            """
            matched_pre_objs + missed_pre_objs + ignore_pre_track_objs = pre_track_objs
            matched_gt_objs + missed_gt_objs + ignore_gt_track_objs = gt_track_objs
            len(matched_pre_objs) == len(matched_gt_objs)
            """ 
            valid_pre_track_objs = matched_pre_objs + missed_pre_objs

            # 更新leaderboard
            self.leader_board[video_id]["GT_total"] = len(valid_gt_track_objs)
            self.leader_board[video_id]["PRE_total"] = len(valid_pre_track_objs)
            self.leader_board[video_id]["HIT"] = len(matched_pre_objs)

            # 更新register
            self.register[video_id]['TP'] = matched_pre_objs
            self.register[video_id]['FP'] = missed_pre_objs
            self.register[video_id]['FN'] = missed_gt_objs
            self.register[video_id]['IGNORE'] = ignore_gt_track_objs
            
     
        # 增加OverAll指标 
        self.leader_board["OVERALL"] = {}
        self.leader_board["OVERALL"]["HIT"] = sum([self.leader_board[video_id]["HIT"] for video_id in self.video_id_list])
        self.leader_board["OVERALL"]["PRE_total"] = sum([self.leader_board[video_id]["PRE_total"] for video_id in self.video_id_list])
        self.leader_board["OVERALL"]["GT_total"] = sum([self.leader_board[video_id]["GT_total"] for video_id in self.video_id_list])

        # 更新看板计算准召冗余
        for video_id in self.video_id_list+["OVERALL"]:
            #准确部分 
            if self.leader_board[video_id]["PRE_total"] == 0:
                self.leader_board[video_id]["Precision"] = 1
            else:
                if self.leader_board[video_id]["HIT"] > 0:
                    self.leader_board[video_id]["Precision"] = self.leader_board[video_id]["HIT"] / self.leader_board[video_id]["PRE_total"]

            #召回部分
            if self.leader_board[video_id]["GT_total"] == 0:
                self.leader_board[video_id]["Recall"] = 1
            else:
                if self.leader_board[video_id]["HIT"] > 0:
                    self.leader_board[video_id]["Recall"] = self.leader_board[video_id]["HIT"] / self.leader_board[video_id]["GT_total"]

            if self.leader_board[video_id]["HIT"] > 0:
                self.leader_board[video_id]["F1"] = 2 * self.leader_board[video_id]["Precision"] * self.leader_board    [video_id]["Recall"] / (self.leader_board[video_id]["Precision"] + self.leader_board[video_id]["Recall"] + 0.00001)

    def eval_video(self, gt_track_objs, pre_track_objs, ignore_mode=False):
        """单个video维度""" 
        num_gt = len(gt_track_objs)
        num_pre = len(pre_track_objs)
        gt_match_mat = np.zeros(num_gt, np.int8)
        pre_match_mat = np.zeros(num_pre, np.int8)
        
        # 初始时间矩阵、空间矩阵、相似度矩阵
        tiou_mat = np.zeros((num_gt, num_pre))
        piou_mat = np.zeros((num_gt, num_pre))
        
        for i in range(num_gt):
            for j in range(num_pre):
                gt_object = gt_track_objs[i]
                pre_object = pre_track_objs[j]
                gt_frame_ids = set(gt_object['tracks'].keys())
                pre_frame_ids = set(pre_object['tracks'].keys())
                if not ignore_mode:
                    tiou_mat[i][j] = len(gt_frame_ids & pre_frame_ids) / len(gt_frame_ids | pre_frame_ids)                
                else:
                    if len(gt_frame_ids & pre_frame_ids) > 0:
                        tiou_mat[i][j] = 1.0
                # NOTE 节省计算时间///
                if tiou_mat[i][j] >= self.tiou_thresh:
                    piou_mat[i][j] = self.cal_piou(gt_object['tracks'], pre_object['tracks'])

        # 主匹配逻辑
        for i in range(num_gt):
            for j in range(num_pre):
                if not ignore_mode:
                    if gt_match_mat[i] == 0 and pre_match_mat[j] == 0:
                        if tiou_mat[i][j] > self.tiou_thresh and piou_mat[i][j] > self.piou_thresh:
                            # 匹配成功
                            gt_match_mat[i] = 1
                            pre_match_mat[j] = 1
                else: # ignore_mode gt可以ignore掉多个pre
                    if pre_match_mat[j] == 0:
                        if tiou_mat[i][j] > self.tiou_thresh and piou_mat[i][j] > self.piou_thresh:
                            # 匹配成功
                            gt_match_mat[i] = 1
                            pre_match_mat[j] = 1
        
        # 处理返回         
        matched_gt_objs = []
        matched_pre_objs = []
        missed_gt_objs = []
        missed_pre_objs = []

        for i in range(num_gt):
            if gt_match_mat[i] == 1:
                matched_gt_objs.append(gt_track_objs[i])
            else:
                missed_gt_objs.append(gt_track_objs[i])
        for j in range(num_pre):
            if pre_match_mat[j] == 1:
                matched_pre_objs.append(pre_track_objs[j])
            else:
                missed_pre_objs.append(pre_track_objs[j])
        return matched_gt_objs, matched_pre_objs, missed_gt_objs, missed_pre_objs

    def cal_piou(self, frame_dict1, frame_dict2):
        inter_ids = frame_dict1.keys() & frame_dict2.keys()
        piou = 0
        for frame_id in inter_ids:        
            box1 = frame_dict1[frame_id]
            box2 = frame_dict2[frame_id]
            piou += self.cal_iou_poly(box1[:8], box2[:8])
        return piou / len(inter_ids)

    def cal_similarity(self, string1, string2):
        if string1 == "#null" or string2 == '#null':
            return 1.0
        string1= self.strip_points(string1)
        string2= self.strip_points(string2)
        #在去掉标点以后再判断字符串是否为空, 防止标点字符串导致下面分母为0 
        if string1 == "" and string2 == "":
            return 1.0
        # TODO 确定是否保留，当字符串差1个字符的时候，也算对
        if Levenshtein.distance(string1, string2) == 1 :
            return 0.95
        return 1 - Levenshtein.distance(string1, string2) / max(len(string1), len(string2))
    
    @staticmethod            
    def cal_iou_poly(bbox1, bbox2):
        '''
        凸四边形的iou计算
        :param bbox1: [x1, y1, x2, y2, x3, y3, x4, y4]
        :param bbox2:[x1, y1, x2, y2, x3, y3, x4, y4]
        :return:
        '''
        bbox1 = np.array([bbox1[0], bbox1[1],
                        bbox1[6], bbox1[7],
                        bbox1[4], bbox1[5],
                        bbox1[2], bbox1[3]]).reshape(4, 2)
        poly1 = Polygon(bbox1).convex_hull   # 左上 左下 右下 右上 左上
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

    @staticmethod
    def strip_points(s):
        s = str(s)
        for i in string.punctuation + punctuation:
            s = s.replace(i, '')
        s = s.replace(' ', '')
        s = s.upper()
        return s


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="视频OCR评估指标计算代码")
    parser.add_argument("--pre_path", dest="pre_path", type=str, help="模型输出结果文件路径")
    parser.add_argument("--gt_path", dest="gt_path", type=str, help="标注数据文件路径")
    parser.add_argument("--data_type", dest="data_type", type=str, default='videotext', help="support videotext")
    parser.add_argument("--frame_dir", dest="frame_dir", type=str, default='data/test_5947/', help="original video frame images path")
    parser.add_argument("--out_dir", dest="out_dir", type=str, default='data/out_dir/', help="output vis dir")
    parser.add_argument("--vis", dest="vis", action='store_true', default=False, help="visulization video results")
    parser.add_argument("--det_only", dest="det_only", action="store_true", default=False, help="是否则仅评测跟踪指标")
    args = parser.parse_args()
    
    evaler = Evaluator(pre_path=args.pre_path, 
                       gt_path=args.gt_path, 
                       data_type=args.data_type,
                       det_only=args.det_only)
    with open('./res.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(evaler.leader_board, indent=4, sort_keys=True))
    print(json.dumps(evaler.leader_board, indent=4, sort_keys=True))
    