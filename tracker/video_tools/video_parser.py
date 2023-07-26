#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   video_tool.py
@Time    :   2021/04/22 17:17:38
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
import os
from os import path as osp
import cv2
import os
import os.path as osp
from tqdm import tqdm
import sys
sys.path.append(osp.dirname(__file__))
from visulization import plot_tracking
from glob import glob
import shutil


class VideoParser(object):
    """
    @description  : Video表示类，用于记录video的信息
    ---------
    """
    def __init__(self, video_path, verbose=False):
        print(video_path)

        assert osp.exists(video_path)
        video_name = osp.basename(video_path)
        self.id = video_name.split('.')[0]
        self.capture = cv2.VideoCapture(video_path)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.verbose = verbose
        if not self.verbose:
            print("video_id: {}, fps: {}, video_width: {}, video_height: {}".format(self.id, self.fps, self.width, self.height))
    
    def to_imgs(self, out_path, cap_ps=None):
    
        save_dir = out_path    # osp.join(out_path, self.id)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        
        # 从第0帧开始，与快手的抽帧命名保持一致
        frame_num = 1
        save_frame_num = 0
        phase = 1
        if cap_ps and cap_ps < self.fps:
            phase = (self.fps+cap_ps-1) // cap_ps
        success, frame = self.capture.read()
        while success:
            if not cap_ps or frame_num % phase == 1:
                cv2.imwrite(osp.join(save_dir, str(frame_num)+'.jpg'), frame)
                save_frame_num += 1
            success, frame = self.capture.read()
            frame_num += 1
        if not self.verbose:
            print("total frames: {}, save frames: {}".format(frame_num, save_frame_num))


class VideoTool(object):
    """
    视频工具类
    """
    def __init__(self):
        pass

    @staticmethod
    def video2imgs(video_path, out_path, cap_ps=None, verbose=False):
        videoParser = VideoParser(video_path, verbose=verbose)
        videoParser.to_imgs(out_path, cap_ps)

    @staticmethod
    def imgs2video(frame_dir, video_path, rate=5):
        """
        @description  :
        ---------
        @param  : rate 合成1s视频需要的图片数
        -------
        @Returns  :
        -------
        """
        output_dir = osp.dirname(video_path)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        tmp_dir = './tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        img_list = sorted(glob(osp.join(frame_dir, "*.jpg")), key=lambda x: int(osp.basename(x).split(".")[0]))
        frame_id = 0
        for img_path in img_list:
            shutil.copy(img_path, osp.join(tmp_dir, str(frame_id)+".jpg"))
            frame_id += 1

        
        cmd_str = 'ffmpeg -f image2 -r {} -i {}/%d.jpg  {}'.format(rate, tmp_dir, video_path)
        os.system(cmd_str)

    @staticmethod
    def frames_tracking(img_dir, out_dir, track_list, frame_size=None, keyword=""):
        """
        @description  : 将图片绘制成跟踪轨迹
        ---------
        @param  :
            img_dir: 原始图片文件夹
            out_dir: 存储绘制后的图片
            track_dict: 该video中以track_id为key的字典
        -------
        @Returns  :
        -------
        """
        # 读取label 转化为以帧为单位
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        label_dict = {}
        # 格式修改，转化为以frame_id为key的代码
        if isinstance(track_list, list):
            for track_id, item in enumerate(track_list):
                frame_dict = item['tracks']
                text = item['text']
                for frame_id, box in frame_dict.items():
                    content = "###"
                    if len(box) == 8:
                        x1, y1, x2, y2, x3, y3, x4, y4 = box
                    else:
                        x1, y1, x2, y2, x3, y3, x4, y4, content = box

                    if frame_id not in label_dict:
                        label_dict[frame_id] = []
                    label_dict[frame_id].append([track_id, x1, y1, x2, y2, x3, y3, x4, y4, text])
        else:
            for track_id, item in track_list.items():
                frame_dict = item['tracks']
                text = item['text']
                for frame_id, box in frame_dict.items():
                    x1, y1, x2, y2, x3, y3, x4, y4, content = box
                    if frame_id not in label_dict:
                        label_dict[frame_id] = []
                    label_dict[frame_id].append([track_id, x1, y1, x2, y2, x3, y3, x4, y4, text])
   
        # 开始画图
        for img_path in glob(osp.join(img_dir, "*.jpg")):
            img_name = osp.basename(img_path)
            frame_id = img_name.split('.')[0]
            # 没有检测结果，直接cp
            if frame_id not in label_dict:
                track_list = []
            else:
                track_list = label_dict[frame_id]
            plotted_img_path = osp.join(out_dir, img_name)
            img = cv2.imread(img_path)
            if frame_size:
                img = cv2.resize(img, frame_size)
            points = []
            content_list = []
            track_id_list = []
            for track_item in track_list:
                points.append(track_item[1:9])
                content_list.append(track_item[9])
                track_id_list.append(track_item[0])
            plotted_img = plot_tracking(img, points, content_list, track_id_list, frame_id=frame_id, keyword=keyword)
            cv2.imwrite(plotted_img_path, plotted_img)


if __name__ == "__main__":

    # video_dir = '/share/lizhuang05/datasets/ICDAR2013_VIDEO/ch3_train/'
    # 视频转图片
    # for video_path in glob(os.path.join(video_dir, "*.mp4")):
    video_path = '/share/lizhuang05/tmp/35669385088.mp4'
    videoTool = VideoTool()
    videoTool.video2imgs(video_path=video_path, out_path='/share/lizhuang05/tmp/', cap_ps=5)

    # # tracking结果可视化
    # videoTool.frames_tracking(img_dir='./data/47400793724', out_dir='./data/47400793724_pre')
    # frame可视化转视频
    # videoTool.imgs2video(frame_dir='./data/47400793724_pre', video_path='./data/47400793724.mp4')
