#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   track.py
@Time    :   2021/06/01 14:51:41
@Author  :   lzneu
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
from collections import OrderedDict

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
            ICDAR21_DetectionTracks[frame].append({"points":[str(i) for i in points],"ID":str(track_id),"transcription":str(text)})
            
            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(track_id))
            object1.setAttribute("Transcription", str(text))
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


def demo(model, config, frame_dir, save_images=False, save_videos=False):

#     vis_dir = osp.join(out_dir, 'frame')
#     result_file_path = os.path.join(out_dir, "result" + '.json')
#     mkdir_if_missing(out_dir)
#     mkdir_if_missing(vis_dir)
    frame_info_list = []
    # 获取单帧信息（图像OCR结果）
    # 单帧进行识别
    for img_path in tqdm(glob(osp.join(frame_dir, "*.jpg"))):
        frame_id = osp.basename(img_path).split('.')[0]
        frame_info,_ = model.predict(img_path)
        frame_info['frame_id'] = str(int(frame_id))
        frame_info_list.append(frame_info)

    # 排序
    frame_info_list = sorted(frame_info_list, key=lambda x: int(x['frame_id']))

    # 执行跟踪
    re_results = track_online(config['tracker'], frame_info_list)

    result_dict = {}
    for frame_id in range(len(frame_info_list)):
        frame_id= frame_id+1
        
        if str(frame_id) not in re_results:
            result_dict[str(frame_id)] = []
            pass
        else:
            lines = re_results[str(frame_id)]
            result_dict[str(frame_id)] = lines

    return result_dict


def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def write_lines(p, lines):
    p = get_absolute_path(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            f.write(line)

def make_parent_dir(path):
    """make the parent directories for a file."""
    parent_dir = get_dir(path)
    mkdir(parent_dir)

def exists(path):
    path = get_absolute_path(path)
    return os.path.exists(path)

def mkdir(path):
    """
    If the target directory does not exists, it and its parent directories will created.
    """
    path = get_absolute_path(path)
    if not exists(path):
        os.makedirs(path)
    return path

def get_dir(path):
    '''
    return the directory it belongs to.
    if path is a directory itself, itself will be return
    '''
    path = get_absolute_path(path)
    if is_dir(path):
        return path;
    return os.path.split(path)[0]

def is_dir(path):
    path = get_absolute_path(path)
    return os.path.isdir(path)


def track(model, data_root, config, save_images=False, save_videos=False):
    dataset_result = {}
    seqs = os.listdir(data_root)

        
    for seq in tqdm(seqs):
        print("跟踪{}中".format(seq))
        frame_dir = osp.join(data_root, seq)
        if not os.path.isdir(frame_dir):
            continue
#         output_dir = osp.join(out_dir, seq)
#         mkdir_if_missing(output_dir)
        seq_results = demo(model, config,
                            frame_dir,
                            save_images=save_images,
                            save_videos=save_videos)
        dataset_result[seq] = seq_results

    for video_name in dataset_result:
        annotation_one = dataset_result[video_name]

        xml_name = video_name.split("_")
        xml_name = xml_name[0] + "_" + xml_name[1]
#         xml_name = video_name.replace("/","_")
    
        predict_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/xml_spot","res_{}.xml".format(xml_name.replace("V","v")))
        json_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/json_spot","{}.json".format(video_name))
        
#         predict_path = os.path.join("./outputs/pan_pp_r18_minetto_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_minetto_desc/json","{}.json".format(video_name))

#         predict_path = os.path.join("./outputs/pan_pp_r18_YVT_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_YVT_desc/json","{}.json".format(video_name))

#         predict_path = os.path.join("./outputs/pan_pp_r18_BOVText_desc/xml","res_{}.xml".format(xml_name.replace("V","v")))
#         json_path = os.path.join("./outputs/pan_pp_r18_BOVText_desc/json","{}.json".format(xml_name))
        
        Generate_Json_annotation(annotation_one,json_path,predict_path)

def getBboxesAndLabels_icd131(annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    Transcriptions = []
    IDs = []
    rotates = []
    confidences = []
    # points_lists = [] # does not contain the ignored polygons.
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))         
        IDs.append(annotation.attrib["ID"])
        Transcriptions.append(annotation.attrib["Transcription"])
#         confidences.append(annotation.attrib["confidence"])
        confidences.append(1)
        bboxes.append(points)

    if bboxes:
        IDs = np.array(IDs, dtype=np.int64)
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 8), dtype=np.float32)
        IDs = np.array([], dtype=np.int64)
        Transcriptions = []
        confidences = []
        
    return bboxes, IDs, Transcriptions, confidences

def parse_xml_rec(annotation_path):
    utf8_parser = ET.XMLParser(encoding='gbk')
    with open(annotation_path, 'r', encoding='gbk') as load_f:
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()  # 获取树型结构的根
    
    ann_dict = {}
    for idx,child in enumerate(root):
#         image_path = os.path.join(video_path, child.attrib["ID"] + ".jpg")

        bboxes, IDs, Transcriptions, confidences = \
            getBboxesAndLabels_icd131(child)
        ann_dict[child.attrib["ID"]] = [bboxes,IDs,Transcriptions,confidences]
    return ann_dict

def getid_text():
    new_xml_dir_ = "./outputs/pan_pp_r18_ICDAR15/xml_spot"
    
    
    voc_dict = {"res_video_11.xml": "Video_11_4_1_GT_voc.txt", "res_video_15.xml": "Video_15_4_1_GT_voc.txt", "res_video_17.xml": "Video_17_3_1_GT_voc.txt", "res_video_1.xml": "Video_1_1_2_GT_voc.txt", "res_video_20.xml": "Video_20_5_1_GT_voc.txt", "res_video_22.xml": "Video_22_5_1_GT_voc.txt", "res_video_23.xml": "Video_23_5_2_GT_voc.txt", "res_video_24.xml": "Video_24_5_2_GT_voc.txt", "res_video_30.xml": "Video_30_2_3_GT_voc.txt", "res_video_32.xml": "Video_32_2_3_GT_voc.txt", "res_video_34.xml": "Video_34_2_3_GT_voc.txt", "res_video_35.xml": "Video_35_2_3_GT_voc.txt", "res_video_38.xml": "Video_38_2_3_GT_voc.txt", "res_video_39.xml": "Video_39_2_3_GT_voc.txt", "res_video_43.xml": "Video_43_6_4_GT_voc.txt", "res_video_44.xml": "Video_44_6_4_GT_voc.txt", "res_video_48.xml": "Video_48_6_4_GT_voc.txt", "res_video_49.xml": "Video_49_6_4_GT_voc.txt", "res_video_50.xml": "Video_50_7_4_GT_voc.txt", "res_video_53.xml": "Video_53_7_4_GT_voc.txt", "res_video_55.xml": "Video_55_3_2_GT_voc.txt", "res_video_5.xml": "Video_5_3_2_GT_voc.txt", "res_video_6.xml": "Video_6_3_2_GT_voc.txt", "res_video_9.xml": "Video_9_1_1_GT_voc.txt"}
    
    for xml in tqdm(os.listdir(new_xml_dir_)):
        id_trans = {}
        id_cond = {}
        if ".txt" in xml or "ipynb" in xml:
            continue
                
        lines = []
        xml_one = os.path.join(new_xml_dir_,xml)
        ann = parse_xml_rec(xml_one)
        for frame_id_ann in ann:
            points, IDs, Transcriptions,confidences = ann[frame_id_ann]
            for ids, trans, confidence in zip(IDs,Transcriptions,confidences):
                if str(ids) in id_trans:
                    id_trans[str(ids)].append(trans)
                    id_cond[str(ids)].append(float(confidence))
                else:
                    id_trans[str(ids)]=[trans]
                    id_cond[str(ids)]=[float(confidence)]
                    
        id_trans = sort_key(id_trans)
        id_cond = sort_key(id_cond)
#         print(xml)
        for i in id_trans:
            txts = id_trans[i]
            confidences = id_cond[i]
            txt = max(txts,key=txts.count)
            lines.append('"'+i+'"'+","+'"'+txt+'"'+"\n")
        write_lines(os.path.join(new_xml_dir_,xml.replace("xml","txt")),lines)
        
def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

# 普通 dict 插入元素时是无序的，使用 OrderedDict 按元素插入顺序排序
# 对字典按key排序, 默认升序, 返回 OrderedDict
def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict


if __name__ == '__main__':
    from tracker.video_tools import evaluation

    ids = 'online_config_601_5fps'
    config_path = './config/CoText_r18_ic15_desc.py'
    checkpoint_path = './outputs/CoText_r18_ic15_desc/51_321_0_0_0_checkpoint.pth.tar'# 3_162_0_0_0_checkpoint.pth.tar' 
    data_root=  '/share/wuweijia/Data/ICDAR2015_video/test/frames'
    
#     data_root = "/home/wangjue_Cloud/wuweijia/Data/VideoText/minetto/minetto_test"
#     data_root = "/home/wangjue_Cloud/wuweijia/Data/VideoText/YVT/YVT_test"
#     data_root = "/share/wuweijia/MyBenchMark/MMVText/BOVTextV2/Test/Frames"
    

    pANppE2E = PANppE2E(checkpoint_path, config_path, ctc=True)
    track(pANppE2E, data_root, config,
                    save_images=False,
                    save_videos=False)
    getid_text()



