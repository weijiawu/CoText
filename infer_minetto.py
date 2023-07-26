#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   infer.py
@Time    :   2021/08/09 17:16:57
@Author  :   weijia
@Version :   1.0
@Contact :   weijiawu@zju.edu.cn
@License :   (C)Copyright 2021-2022, Zhejiang University
@Desc    :   改进PANPP-视频OCR
'''
import torchvision.transforms as transforms
from PIL import Image
from models import build_model
from mmcv import Config
from glob import glob
import torch
import numpy as np
import cv2
from mmcv import Config
from dataset.dataset_tool import get_vocabulary
from models.utils import fuse_module
from os import path as osp
from tracker.video_tools import visulization as vis
# here put the import lib
from tqdm import tqdm
import time
# import mxnet as mx
import os

    
class PANppE2E(object):
    def __init__(self, checkpoint_path, config_path, ctc=False):
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=ctc)  #CHINESE  LOWERCASE
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

        frame_id = osp.basename(img_path).split('.')[0]
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]

#         img = mx.image.imdecode(open(img_path,'rb').read())
#         img = mx.image.imread(img_path)
#         img = np.array(img)
        
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
#         print("preprocess:",time.time()-start)
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model(**data)
#         print("model:",time.time()-start)
        
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
#         res['out_rec'] = np.zeros((len(valid_indexs), 38,32))

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


if __name__ == '__main__':
    config_path = 'config/pan_pp/pan_pp_r18_minetto_desc.py'
    checkpoint_path = './outputs/pan_pp_r18_ic15_detrec/24_0_0_0_0_checkpoint.pth.tar'
#     img_path = '/share/wuweijia/Data/ICDAR2013_video/test/frames/'
#     img_path = "/home/wangjue_Cloud/wuweijia/Data/ICDAR2013_video/test/frames"
#     img_path = '/home/wangjue_Cloud/wuweijia/Data/VideoText/YVT/YVT_test'
    img_path = "/home/wangjue_Cloud/wuweijia/Data/VideoText/minetto/minetto_test"
    
    model = PANppE2E(checkpoint_path, config_path, ctc=True)
    show = False
    import time
    start = time.time()
    image_len = 0
    rec_head_cost = 0
    backbone_time = 0
    neck_time = 0
    det_head_time = 0
    desc_time = 0
    print("ssss")
    for cls in tqdm(os.listdir(img_path)):
        if ".xml" in cls or ".txt" in cls:
            continue
        cls_path = os.path.join(img_path,cls)
        image_len += len(os.listdir(cls_path))
        for image_ in tqdm(os.listdir(cls_path)):
#             if "jpg" in image_:
            image_path_ = os.path.join(cls_path,image_)
            res,outputs = model.predict(image_path_)

            rec_head_cost+= outputs["rec_time"]
            backbone_time+= outputs["backbone_time"]
            neck_time+= outputs["neck_time"]
            det_head_time+= outputs["det_head_time"]
            desc_time+= outputs["desc_time"]
            new_ = image_.replace(".jpg","")[-4:]
            res_path = "./eval/Evaluation_Detection_Minetto/icdar15_evaluate/res/" + "{}_GT.json{}.txt".format(cls,int(new_))
#             res_path = "./eval/Evaluation_Detection_YVT_video/icdar15_evaluate/res/" + "{}.json{}.txt".format(cls,int(image_.replace(".jpg","").split("_")[-1][-4:]))
#             print(res_path)
    
            vis.write_detections(res['dets'], res['contents'], res['word_scores'],res_path)

            if show and res['dets'].any():
                plt_img,hotmap = vis.plot_detections(cv2.imread(image_path_), res['dets'], res['contents'], res['word_scores'])
                cv2.imwrite('./outputs/demo/show/{}'.format(image_), plt_img)
                cv2.imwrite('./outputs/demo/show/hot{}'.format(image_), hotmap)
    print("time cost:",time.time() - start)
    print("image number:",image_len)
    print("rec_head_cost cost:",rec_head_cost)
    print("backbone_time cost:",backbone_time)
    print("neck_time cost:",neck_time)
    print("det_head_time cost:",det_head_time)
    print("desc_time cost:",desc_time)
    
#     print(res)
