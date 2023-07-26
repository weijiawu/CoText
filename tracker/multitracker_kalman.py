#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multitracker.py
@Time    :   2021/07/13 17:03:14
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

import sys
import os
import os.path as osp
sys.path.append(osp.dirname(__file__))
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import numpy as np
from collections import deque
import itertools
import time
import torch
import cv2
import Levenshtein
from tools.common import logger
import matching
from basetrack import BaseTrack, TrackState
from kalman_filter import KalmanFilter
import copy
from glob import glob
import re


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, bbox, score, content, temp_feat=None, buffer_size=30, perform_kalman=False):
#         self.shared_kalman = KalmanFilter()
        # wait activate
        self._bbox = np.asarray(bbox, dtype=np.float)
        self.is_activated = False
        self.frame_id_list = []
        self.score = score
        self.content = self.deal_content(content)    # TODO 这里处理成形式
        self.tracklet_len = 0   # 轨迹所占的连续帧的数量

        self.smooth_feat = None
        self.update_features(temp_feat)
        # self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.5

        # 卡尔曼滤波
        self.perform_kalman = perform_kalman
        if self.perform_kalman:
            self.kalman_filter = None
            self.mean, self.covariance = None, None


    def update_features(self, feat):
        if feat is None:
            self.curr_feat = None
            return None
        # feat /= np.linalg.norm(feat)   # 求二范数 这里就是做了归一化
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        # self.features.append(feat)
        # self.smooth_feat /= np.linalg.norm(self.smooth_feat)


    def activate(self, frame_id, track_id, kalman_filter=None):   # 开启一个新的轨迹
        """Start a new tracklet"""
        self.track_id = track_id # self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if self.perform_kalman:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(self._bbox)  # 更新轨迹的均值方差

        # if frame_id == 0:
        #     self.is_activated = True
        self.is_activated = True    # 已确认就是这里的问题，只有在第一帧进行了activate  作者认为只有连续出现两帧才算一个track
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.frame_id_list.append((frame_id, self.bbox, self.content))     # 记录该轨迹的frame_id，离线使用，从0开始

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.perform_kalman:
            self.mean, self.covariance = self.kalman_filter.update(
                                              self.mean, self.covariance, new_track.bbox)
        self.update_features(new_track.curr_feat)
        self._bbox = new_track.bbox
        self.content = new_track.content
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.frame_id_list.append((frame_id, self.bbox, self.content))

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self._bbox = new_track.bbox
        self.content = new_track.content
        self.score = new_track.score
        self.frame_id_list.append((frame_id, self.bbox, self.content))
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        if self.perform_kalman:
            self.mean, self.covariance = self.kalman_filter.update(
                                                self.mean, self.covariance, new_track.bbox)
        if update_feature:
            self.update_features(new_track.curr_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)


    # @jit(nopython=True)
    @property
    def bbox(self):   # 最终的track是通过这个方法生成的，因此是平滑的
        """
        Get current position in bounding box format
        """
        if self.perform_kalman and self.mean is not None:
            ret = self.mean[:8].copy()
            return ret
        return self._bbox.copy()
        
    # 名称表示
    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def deal_content(content):
        content = content.replace('UNK', '')
        content = re.sub(u'[\\\!\s#$%&\'()*+,\-./;<=>?@[\]^_{|}~—‘’“”…、。《》【】，！：；:？]', u'', content)
        content = content.upper()
        return content


class Tracker(object):
    def __init__(self, config):
        self.config = config
        self.tracked_stracks = []       # type: list[STrack]
        self.lost_stracks = []          # type: list[STrack]
        self.removed_stracks = []       # type: list[STrack]
        self.frame_id = -1
        self.track_count = 0     # 这里是重置track_id
        self.max_time_lost = config["max_time_lost"]   # 最大丢失时间，max_time_lost帧 未出现就认为是下一个目标，因此需要设置
        if self.config["perform_kalman"]:
            self.kalman_filter = KalmanFilter()
#             self.mean = np.array(self.config["mean"], dtype=np.float32).reshape(1, 1, 3)
#             self.std = np.array(self.config["std"], dtype=np.float32).reshape(1, 1, 3)

 
    # 新增方法，用于重新初始化状态，无需再次加载模型
    def reset_state(self):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = -1
        self.track_count = 0     # 这里是重置track_id
        if self.config["perform_kalman"]:
            self.kalman_filter = KalmanFilter()
#             self.mean = np.array(self.config["mean"], dtype=np.float32).reshape(1, 1, 3)
#             self.std = np.array(self.config["std"], dtype=np.float32).reshape(1, 1, 3)

    # 获得最终的输出，这里获得后需要调用reset_state方法来重置tracker
    # 在离线时使用
    def get_tracks(self):
        output_stracks = []
        for track in self.tracked_stracks + self.lost_stracks + self.removed_stracks:
            if not track.is_activated:
                continue
            output_stracks.append(track)
        return output_stracks

    def update(self, dets, contents, id_features):
        self.frame_id += 1
#         print('===========Frame {}=========='.format(self.frame_id))
        
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
#         print("检测输出{}个".format(dets.shape[0]))
        
        ''' Step 1: 初始化轨迹'''
        if len(dets) > 0:
            '''Detections''' # 这里将检测结果表示为单个跟踪对象
            detections = [STrack(bbox[:8], bbox[8], content, f, perform_kalman=self.config["perform_kalman"]) for
                          (bbox, content, f) in zip(dets[:, :9], contents, id_features)]
        else:
            detections = []
        
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []   # 存储本次未被激活的轨迹
        tracked_stracks = []  #  存储本次状态为激活的轨迹
        for track in self.tracked_stracks: 
            if not track.is_activated:
                raise "Not a valid Track, please check the track {}".format(str(track))
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)   # 轨迹池由激活轨迹和lost轨迹构成
#         print("轨迹池数量: {}".format(len(strack_pool)))
        if self.config["perform_kalman"]:
            STrack.multi_predict(strack_pool)   # 更新这些track对象的均值标准差
        
        ''' Step 2: First association, with embedding'''
        matched_detection_index = []
        matched_track_index = []
        if self.config["perform_reid"]:
            dists1 = matching.embedding_distance(strack_pool, detections)    # 计算现存激活轨迹和检测构成的单个轨迹是否能够匹配，余弦距离
#             print("|".join([s.content for s in strack_pool]))
#             print("|".join([s.content for s in detections]))
#             print(dists1)
            if self.config["perform_kalman"]:
                dists1 = matching.fuse_motion(self.kalman_filter, dists1, strack_pool, detections)   # TODO 这里是卡尔曼滤波 对距离做了一些加强 看下是否去掉
            matches, u_track, u_detection = matching.linear_assignment(dists1, thresh=self.config["embedding_distance_thresh"])   # 匹配结果、未匹配轨迹、未匹配检测
            for itracked, idet in matches:
                # 判断字符串是否一致
                track = strack_pool[itracked]
                det = detections[idet]
                matched_track_index.append(int(itracked))
                matched_detection_index.append(int(idet))
                
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)    # 将当前帧检测更新到该轨迹中，embedding 特征使用0.9的滑动平均来更新
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in range(len(detections)) if i not in matched_detection_index]    # 只处理未匹配的结果
        # NOTE 这里更新轨迹的时候，只取了激活状态的路径，即认为lost状态的轨迹，不能通过IOU来匹配
        r_tracked_stracks = [strack_pool[i] for i in range(len(strack_pool)) if strack_pool[i].state == TrackState.Tracked and i not in matched_track_index]
        matched_detection_index = []
        matched_track_index = []
        if self.config["perform_iou"]:
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.config["IOU_distance_thresh"]) # IOU > 0.7认为是匹配
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if Levenshtein.ratio(track.content, det.content) < self.config["IOU_similarity_thresh"] and (not det.content.startswith(track.content)) and (not track.content.startswith(det.content)) and Levenshtein.distance(track.content, det.content) >= self.config['min_char_tolerance']:
#                     print("=".join(["字符串相似度低 IOU不合并", str(dists[itracked][idet]), track.content, det.content, str(Levenshtein.ratio(track.content, det.content))]))
                    continue
#                 print("=".join(["IOU合并", str(dists[itracked][idet]), track.content, det.content, str(Levenshtein.ratio(track.content, det.content))]))
                matched_track_index.append(int(itracked))
                matched_detection_index.append(int(idet))
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        # 更新lost
        for i in range(len(r_tracked_stracks)):
            if i in matched_track_index:
                continue
            track = r_tracked_stracks[i]
            if not track.state == TrackState.Lost:  # 当前帧没有匹配到检测结果，放到lost_stracks里面，并标记为lost
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 4: Init new stracks"""  # 将未匹配到的检测结果初始化成一个轨迹
        for i in range(len(detections)):
            if i in matched_detection_index:
                continue
            track = detections[i]
            if self.config["perform_kalman"]:
                track.activate(self.frame_id, self.track_count,kalman_filter=self.kalman_filter)
                self.track_count += 1
                activated_starcks.append(track)   # 此次新增的轨迹
            else:
                track.activate(self.frame_id, self.track_count)
                self.track_count += 1
                activated_starcks.append(track)   # 此次新增的轨迹
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:       # 如果超过最大丢失时间(默认1帧)，就removed_stracks进行终止
            if self.frame_id - track.end_frame > self.config["max_time_lost"]: # self.max_time_lost:   
                track.mark_removed()   # self.state = TrackState.Removed 标记为remove
                removed_stracks.append(track)

        """ 更新状态，维护self.tracked_stracks、self.lost_stracks、self.removed_stracks"""
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]    # 去掉删除的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)  # 将轨迹增加到新增的轨迹
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)   # refind代表lost过一些帧，又续上了
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)     # 在lost_stracks中去除在这一帧又有匹配结果的对象
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)     # 这里去掉的应该是本轮删除的track, 在lost_stracks中删除removed_stracks的元素
        self.removed_stracks.extend(removed_stracks)                                 # 已经终止的跟踪轨迹
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

#         print('Activated: {}'.format([track.track_id for track in activated_starcks]))
#         print('Refind: {}'.format([track.track_id for track in refind_stracks]))
#         print('Lost: {}'.format([track.track_id for track in lost_stracks]))
#         print('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """删除轨迹a集合中存在于轨迹b集合中的轨迹"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):    # 相距很近的对儿
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:   # 谁短删除谁
            dupb.append(q)   # b中要删除的要删除
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def track_online(config, frame_info_list):
    
    tracker = Tracker(config)
    tracker.reset_state()   # 重置tracker存储的跟踪数据
    results = {}
    frame_id_map = {}

    for index, frame_info in enumerate(frame_info_list):
        frame_id = frame_info['frame_id']
        logger.debug("origin frame_id: {}".format(frame_id))
        dets = frame_info['dets']
        contents = frame_info['contents']
        id_features = frame_info['id_features']
        # run tracking
        tracker.update(dets, contents, id_features)
        frame_id_map[index] =  frame_id

     # 跟踪结束，开始提取结果
    for t in tracker.get_tracks():
        tid = t.track_id
        for frame_item in t.frame_id_list:
            index, bbox, content = frame_item
            if frame_id_map[index] not in results:
                results[frame_id_map[index]] = []
            results[frame_id_map[index]].append((tid, bbox, content))
    
    # 对results进行排序
#     re_results = {}
#     for frame_id, track_list in results.items():
#         results[frame_id] = sorted(track_list, key=lambda x: x[0])
#         for track_item in track_list:
#             track_id, bbox, content = track_item
#             if track_id not in re_results:
#                 re_results[track_id] = {"text": "", "tracks": {}}
#             re_results[track_id]["tracks"][frame_id] = bbox.tolist()  + [content]
    

    return results


if __name__ == '__main__':
    file_path = '/home/lizhuang05/code/video_ocr_e2e/22580022298-1851.npy'
    frame_info_list, config, re_res = np.load(file_path, allow_pickle=True)
    config['max_time_lost'] = 10
    res = track_online(config, frame_info_list)
    # print(res)
    for k, v in re_res.items():
        tracks = v['tracks']
        tracks_offline = res[k]['tracks']
        if len(tracks_offline) != len(tracks):
            frame_id_list = sorted(list(tracks_offline.keys()), key=lambda x: int(x))
            if 60 < int(frame_id_list[0]) < 80:
                print(k, sorted(list(tracks.keys()), key=lambda x: int(x)))
                print(k, sorted(list(tracks_offline.keys()), key=lambda x: int(x)))
            

