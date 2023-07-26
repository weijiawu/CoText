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
from mmcv import Config
from tracker.video_tools import visulization as vis
from tracker.config import config
from glob import glob
from PIL import Image
import pickle
logger.setLevel(logging.INFO)
from collections import OrderedDict
from dataset.dataset_tool import get_vocabulary
from models import build_model
from xml.dom.minidom import Document
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
from tqdm import tqdm
from models.utils import fuse_module
import time
import torchvision.transforms as transforms

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

class PANppE2E(object):
    def __init__(self, checkpoint_path, config_path, ctc=False):
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=ctc)  #CHINESE  LOWERCASE
        print("voc:",len(self.voc))
        self.device = torch.device('cuda')
        self.model =  self.build_model(config_path, checkpoint_path)
        if self.cfg.test_cfg.is_half:
            self.model.half()
        self.model = self.model.to(self.device)
        self.img_size = self.cfg.data.test.short_size
        self.align_mode = self.cfg.data.test.align_mode
        assert self.align_mode in ("short", "long", "force", "online")
        print('align_mode', self.align_mode, self.img_size)
    

    def build_model(self, config_path, checkpoint_path):
        self.cfg = Config.fromfile(config_path)
        for d in [self.cfg, self.cfg.data.test]:
            d.update(dict(
                report_speed=True
            ))
        if hasattr(self.cfg.model, 'recognition_head'):
            self.cfg.model.recognition_head.update(dict(
                voc=self.voc,
                char2id=self.char2id,
                id2char=self.id2char,
            ))
        if hasattr(self.cfg.model, 'description_head'):
            self.cfg.model.description_head.update(dict(
                voc=self.voc
            ))
        model = build_model(self.cfg.model)
        model = model.cuda()
        checkpoint = torch.load(checkpoint_path)
        d = dict()
        for key, value in checkpoint['state_dict'].items():
            if "s_desc" in key or  "s_rec" in key or  "s_det" in key:
                print(key, value)
                continue
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d, strict=False)
        model = fuse_module(model)
        model.eval()
        
        rec_parameter = 0
        for name, param in model.named_parameters():
            if name.split('.')[0] == 'rec_head':
                rec_parameter+=param.numel()
#             print(name)
#             print(param.numel())
        print("rec head parameter:",rec_parameter)
    
        return model

    def preprocess(self, img_path):
        data = {}

#         frame_id = osp.basename(img_path).split('.')[0]
        img = img_path
        img = img[:, :, [2, 1, 0]]
        
        img_meta = dict(
            org_img_size=np.array([img.shape[:2]]))
        if self.align_mode == 'short':
        # img, valid_size = self.scale_aligned_long_padding(img, self.img_size)
            img, valid_size = self.scale_aligned_short(img, self.img_size)
        elif self.align_mode == 'long':
            img, valid_size = self.scale_aligned_long_padding(img, self.img_size)
        elif self.align_mode == 'force':
            img, valid_size = cv2.resize(img, dsize=(self.img_size, self.img_size)), (self.img_size, self.img_size)
        elif self.align_mode == 'online':
            img, valid_size = self.scale_aligned_force_padding(img)
        # img = cv2.resize(img, dsize=(736, 1280))
        # print(img.shape, valid_size)
        img_meta.update(dict(
            img_size=np.array([img.shape[:2]])
        ))
        img_meta.update(dict(
            valid_size=np.array([valid_size])
        ))
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
#         img = img.cuda().unsqueeze(0)
        if self.cfg.test_cfg.is_half:
            img = img.type(torch.HalfTensor).cuda().unsqueeze(0)
        else:
            img = img.cuda().unsqueeze(0)
        data = dict(
            imgs=img,
            img_metas=img_meta,
        )
        data.update(dict(
            cfg=self.cfg
        ))
        
        return data

    def predict(self, img_path):
        start = time.time()
        data = self.preprocess(img_path)
        start = time.time()
        with torch.no_grad():
            outputs = self.model(**data)

        start = time.time()
        struct_outputs = self.post_process(outputs)
#         print("post_process:",time.time()-start)
        start = time.time()
        return struct_outputs,outputs

    def post_process(self, outputs):
        res = {}
        num_dets = len(outputs['bboxes'])
        # valid_indexs = [i for i in range(num_dets)] #  if float(outputs['word_scores'][i]) >  0.4]
#         print(outputs['word_scores'])
        valid_indexs = [i for i in range(num_dets) if float(outputs['word_scores'][i]) >  self.cfg.test_cfg.min_rec_score]

        res['dets'] = np.zeros((len(valid_indexs), 9))
        res['contents'] = [""] * len(valid_indexs)
        res['word_scores'] = [""] * len(valid_indexs)
        res['id_features'] = np.zeros((len(valid_indexs), 128))
        res['out_rec'] = np.zeros((len(valid_indexs), 128,32))
        
        
        for i, valid_index in enumerate(valid_indexs):
            res['dets'][i][:8] = outputs['bboxes'][valid_index]
            res['dets'][i][8] = outputs['scores'][valid_index]
            res['contents'][i] = outputs['words'][valid_index]
            res['word_scores'][i] = float(outputs['word_scores'][valid_index])
            res['id_features'][i] = outputs['multi_info_feature'][valid_index]
#             res['out_rec'][i] = outputs['out_rec'][valid_index]
        return res


    @staticmethod
    def scale_aligned_short(img, short_size=736):
        h, w = img.shape[0:2]
        scale = short_size * 1.0 / min(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 4 != 0:
            h = h + (4 - h % 4)
        if w % 4 != 0:
            w = w + (4 - w % 4)
        if h == img.shape[0] and w == img.shape[1]:
            return img, img.shape[:2]
        img = cv2.resize(img, dsize=(w, h))
        return img, img.shape[:2]

    @staticmethod
    def scale_aligned_long_padding(img, long_size=736):
        h, w = img.shape[0:2]

        scale = long_size * 1.0 / max(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        valid_shape = (h, w)
        img = cv2.copyMakeBorder(img, 0, long_size-h, 0,long_size-w, borderType=cv2.BORDER_CONSTANT, value=(0,))  # top, bottom, left, right
        # print('valid_shape', h, w)
        # print('out_shape', img.shape)
        # cv2.imwrite('./debug.jpg', img)
        return img, valid_shape

    @staticmethod
    def scale_aligned_force_padding(img, short_size=736, long_size=1280):
        h, w = img.shape[0:2]
        scale = short_size * 1.0 / w
        h = int(h * scale + 0.5)
        if h > long_size:
            h = long_size
        w = short_size # int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        # if w % 32 != 0:
        #     w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        valid_shape = (h, w)
        img = cv2.copyMakeBorder(img, 0, long_size-h, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0,))  # top, bottom, left, right
        return img, valid_shape
    
def demo(model, config, video_dir, save_images=False, save_videos=False):
    
    frame_info_list = []
    
    # 抽帧
    frams = []
    video_object = cv2.VideoCapture(video_dir)
    while True:
        ret, frame = video_object.read()
        if ret == False:
            print("extract_frame_from_video(), extract is finished")
            break
        frams.append(frame)
        
    # 单帧进行识别
    for fram_id, img_path in tqdm(enumerate(frams)):
        frame_id = fram_id+1
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
        seq_results = demo(model, config,
                            frame_dir,
                            save_images=save_images,
                            save_videos=save_videos)
        dataset_result[seq.replace(".mp4","")] = seq_results

    for video_name in dataset_result:
        annotation_one = dataset_result[video_name]

        xml_name = video_name.split("_")
        xml_name = xml_name[0] + "_" + xml_name[1]
#         xml_name = video_name.replace("/","_")
    
        predict_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/xml_spot","res_{}.xml".format(xml_name.replace("V","v")))
        json_path = os.path.join("./outputs/pan_pp_r18_ICDAR15/json_spot","{}.json".format(video_name))
        
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
    config_path = './config/pan_pp_r18_ic15_desc.py'
    checkpoint_path = './outputs/pan_pp_r18_ic15_desc/SpottingMOTA0.589IDF10.72.pth.tar'# 3_162_0_0_0_checkpoint.pth.tar' 
    
    video_root=  '/share/wuweijia/Data/ICDAR2015_video/test/video'

    pANppE2E = PANppE2E(checkpoint_path, config_path, ctc=True)
    track(pANppE2E, video_root, config,
                    save_images=False,
                    save_videos=False)
    getid_text()



