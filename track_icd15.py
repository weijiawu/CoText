#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   track.py
@Time    :   2021/06/01 14:51:41
@Author  :   weijia
@Version :   1.0
@Contact :   weijiawu@zju.edu.cn
@License :   (C)Copyright 2021-2022, Zhejiang University
@Desc    :   通过在线获得的OCR结果，进行跟踪视频OCR定位
'''

# here put the import lib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import os.path as osp
import cv2
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
from tracker.multitracker import track_online
from tracker.tools.common import logger, mkdir_if_missing, strip_points
import sys
from tracker.video_tools import visulization as vis
from tracker.config import config
from glob import glob
from PIL import Image
from infer_icd15 import PANppE2E
# from tracker.tools.online_rec import Client
import pickle
logger.setLevel(logging.INFO)
import time

# client = Client( KESS_SERVER_NAME_hori='grpc_mmu_videoOcrRecognitionV6',
#             KESS_SERVER_NAME_ver='grpc_mmu_ocrRecognitionVerticalVideo'
#             )

from xml.dom.minidom import Document
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from tqdm import tqdm


    
class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name,'wb')
        pickle.dump(data_dict,output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            # fp.write(unicode(json.dumps(data_dict, ensure_ascii=False, indent=4) ) )  #可以解决在文件里显示中文的问题，不加的话是 '\uxxxx\uxxxx'
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4) ) )

    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict

def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir,xml_dir_):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1

    doc = Document()
    video_xml = doc.createElement("Frames")

    for frame in TL_Cluster_Video_dict.keys():

        doc.appendChild(video_xml)
        aperson = doc.createElement("frame")
        aperson.setAttribute("ID", str(frame))
        video_xml.appendChild(aperson)

        ICDAR21_DetectionTracks[frame] = []
#         vis_dict[frame_id].append([track_id, bbox[:8], track_dict['text']])

        for text_list in TL_Cluster_Video_dict[frame]:
            track_id, points, text = text_list
            ICDAR21_DetectionTracks[frame].append({"points":[str(i) for i in points],"ID":str(track_id)})

            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(track_id))
            aperson.appendChild(object1)

            for i in range(4):

                name = doc.createElement("Point")
                object1.appendChild(name)
                # personname = doc.createTextNode("1")
                name.setAttribute("x", str(int(points[i*2])))
                name.setAttribute("y", str(int(points[i*2+1])))

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)

    # xml
    f = open(xml_dir_, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()

def get_annotation(video_path):
    annotation = {}
    
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def demo(model, config, frame_dir, dict_cost):

    frame_info_list = []
    # 获取单帧信息（图像OCR结果）
    # 单帧进行识别
    
    for img_path in tqdm(glob(osp.join(frame_dir, "*.jpg"))):
        frame_id = osp.basename(img_path).split('.')[0]
#         annotation = get_annotation("./eval/Evaluation_ICDAR13/gt/{}_GT.json".format(frame_dir.split("/")[-1]))  ,annotation[frame_id]
        frame_info,outputs = model.predict(img_path)
        frame_info['frame_id'] = str(int(frame_id))
        frame_info_list.append(frame_info)
        
        dict_cost["rec_head_cost"]+= outputs["rec_time"]
        dict_cost["backbone_time"]+= outputs["backbone_time"]
        dict_cost["neck_time"]+= outputs["neck_time"]
        dict_cost["det_head_time"]+= outputs["det_head_time"]
        dict_cost["desc_time"]+= outputs["desc_time"]
        dict_cost["det_post_time"] += outputs["det_post_time"]
        dict_cost["number_text"] += outputs["number_text"]
        dict_cost["mask_roi"] += outputs["mask_roi"]
    
    start = time.time()
    # 排序
    frame_info_list = sorted(frame_info_list, key=lambda x: int(x['frame_id']))

    # 执行跟踪
    re_results = track_online(config['tracker'], frame_info_list)
    dict_cost["track_pos_cost"] += time.time() - start
    
    result_dict = {}
    for frame_id in range(len(frame_info_list)):
        frame_id= frame_id+1
        
        if str(frame_id) not in re_results:
            result_dict[str(frame_id)] = []
            pass
        else:
            lines = re_results[str(frame_id)]
            result_dict[str(frame_id)] = lines

    return result_dict,dict_cost





def track(model, data_root, config, save_images=False, save_videos=False):
    dataset_result = {}
    seqs = os.listdir(data_root)
    
    import time
    start = time.time()   
    image_len = 0
    
    dict_cost = {
    "rec_head_cost" : 0,
    "backbone_time" : 0,
    "neck_time" : 0,
    "det_head_time" : 0,
    "desc_time" : 0,
    "track_pos_cost" : 0,
    "det_post_time" : 0,
     "number_text": 0 ,
    "mask_roi": 0
    }
    
    for seq in tqdm(seqs):
#         if seq == "Video_39_2_3":
#             continue
        print("跟踪{}中".format(seq))
        frame_dir = osp.join(data_root, seq)
        if not os.path.isdir(frame_dir):
            continue
        image_len += len(os.listdir(frame_dir))
        
        seq_results,dict_cost = demo(model, config,
                            frame_dir,dict_cost)
        dataset_result[seq] = seq_results
        
    for video_name in dataset_result:
        annotation_one = dataset_result[video_name]
        
        xml_name = video_name.split("_")
        xml_name = xml_name[0] + "_" + xml_name[1]
#         xml_name = video_name.replace("/","_")
        
        predict_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/xml","res_{}.xml".format(xml_name.replace("V","v")))
        json_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/json","{}.json".format(video_name))
        
#         predict_path = os.path.join("./outputs/pan_pp_r18_minetto_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_minetto_desc/json","{}.json".format(video_name))

#         predict_path = os.path.join("./outputs/pan_pp_r18_YVT_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_YVT_desc/json","{}.json".format(video_name))

#         predict_path = os.path.join("./outputs/pan_pp_r18_BOVText_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_BOVText_desc/json","{}.json".format(xml_name))
        
        Generate_Json_annotation(annotation_one,json_path,predict_path)
    print("time cost:",time.time() - start)

    print("image number:",image_len)
    
    print(dict_cost.keys())
    print("backbone_time cost:",dict_cost["backbone_time"])
    print("neck_time cost:",dict_cost["neck_time"])
    print("det_head_time cost:",dict_cost["det_head_time"])
    print("rec_head_cost cost:",dict_cost["rec_head_cost"])
    print("desc_time cost:",dict_cost["desc_time"])
    print("track_pos_cost:",dict_cost["track_pos_cost"])
    print("det_post_time:",dict_cost["det_post_time"])
    print("mask_roi:",dict_cost["mask_roi"])
#     print("number_text:",dict_cost["number_text"])
    
def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    from tracker.video_tools import evaluation

    ids = 'online_config_601_5fps'
    config_path = './config/CoText_r18_ic15_desc.py'
    checkpoint_path = './outputs/CoText_r18_ic15_desc/3_2771_0_0_0_checkpoint.pth.tar'# 3_162_0_0_0_checkpoint.pth.tar' 
    data_root=  '/share/wuweijia/Data/ICDAR2013_video/test/frames'
    
#     data_root = "/home/wangjue_Cloud/wuweijia/Data/VideoText/minetto/minetto_test"
#     data_root = "/home/wangjue_Cloud/wuweijia/Data/VideoText/YVT/YVT_test"
#     data_root = "/share/wuweijia/MyBenchMark/MMVText/BOVTextV2/Test/Frames"
    

    pANppE2E = PANppE2E(checkpoint_path, config_path, ctc=True)
    track(pANppE2E, data_root, config,
                    save_images=False,
                    save_videos=False)

