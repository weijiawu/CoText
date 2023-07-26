#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Date    : 2022/3/29 15:22:13
@Author  : weijiawu
@version : 1.0
@License : (C)Copyright 2021-2022, Zhejiang University
全量数据预测
'''
import os
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import mmcv
import scipy.io as scio
from os import path as osp
import copy
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
from  dataset_tool import *
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
    
torch.manual_seed(23)
torch.cuda.manual_seed(23)
np.random.seed(23)
random.seed(22)


ic15_root_dir = '/share/wuweijia/Data/ICDAR2015_video'


class PAN_IC15_video(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 with_desc=False,
                 read_type='pil',
                 use_ctc=False,
                 direction_aug=False,
                 direction_ratio=0.5,
                 report_speed=False,
                 with_kwaitrain=False):
        self.split = split
        self.is_transform = is_transform
        self.use_ctc = use_ctc
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.for_rec = with_rec
        self.for_desc = with_desc
        self.read_type = read_type
        self.img_paths = {}
        self.gts = {}
        self.texts = {}
        self.img_num = 0
        self.data_list = []  # 最终使用的实际上是这个
        self.direction_aug = direction_aug
        self.direction_ratio = direction_ratio
        self.with_kwaitrain = with_kwaitrain

        # ic15 video
        self.img_paths['ic15'] = []
        self.gts['ic15'] = []
        data_dirs = os.path.join(ic15_root_dir,"train/frames")
        gt_dirs = os.path.join(ic15_root_dir,"train/gt")
        for video_name in os.listdir(data_dirs):
            video_path = os.path.join(data_dirs,video_name)
            img_names = [
                img_name for img_name in mmcv.utils.scandir(video_path, '.jpg')
            ]
            
            ann_path = video_path.replace("frames","gt") + "_GT.xml"
            frame_annotation = self.parse_xml(ann_path)
            
            for idx in range(len(img_names)):
                img_path = os.path.join(video_path,"{}.jpg".format(idx+1))
                self.img_paths['ic15'].append(img_path)
                self.gts['ic15'].append(frame_annotation[str(idx+1)])
        
        self.img_num += len(self.img_paths['ic15'])
        
        
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=self.use_ctc)
        self.max_word_num = 50
        self.max_word_len = 32

        self.data_list = []
        for dataset_cat in self.img_paths.keys():
            dataset_img_list = self.img_paths[dataset_cat]
            dataset_gt_list = self.gts[dataset_cat]
            dataset_cat_list = [dataset_cat]*len(dataset_img_list)
            self.data_list += list(zip(dataset_cat_list,dataset_img_list, dataset_gt_list))
        
        print("training data size: {}".format(len(self.data_list)))
        # 打乱顺序
        # random.shuffle(self.data_list)
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return self.img_num
    
    def parse_xml(self,annotation_path):
        utf8_parser = ET.XMLParser(encoding='gbk')
        with open(annotation_path, 'r', encoding='gbk') as load_f:
            tree = ET.parse(load_f, parser=utf8_parser)
        root = tree.getroot()  # 获取树型结构的根
        return_ann = {}
        for frame_id,child in enumerate(root):
            frame_ann = []
            for annotation in child:
                object_boxes = []
                for point in annotation:
                    object_boxes.append(int(point.attrib["x"]))
                    object_boxes.append(int(point.attrib["y"]))
                quality = annotation.attrib["Quality"]
                Transcription = annotation.attrib["Transcription"]
                if "#" in Transcription:
                    trans = "###"
                else:
                    trans = Transcription
                object_boxes.append(trans)
                frame_ann.append(object_boxes)
            return_ann.update({str(frame_id+1):frame_ann})
        
        return return_ann
    
    # For 自监督
    def __getitem__(self, index): 

        if index == 0:        # 这里需要进行顺序打乱
            print('shuffle traing data!')
            random.shuffle(self.data_list)
        dataset_cat, img_path, gt_path = self.data_list[index]
        
        img, bboxes, words, img_path = self.load_ic15_single(img_path, gt_path)
        
        # 三角度、上下镜像、左右镜像 增强 比例为0.5
        if self.direction_aug and random.random() < self.direction_ratio:
            mode = random.randint(0, 4) 
            # print('mode', mode, img_path)
            img, bboxes, words = random_rot_flip(img, bboxes, words, mode)
        
        h, w = img.shape[0:2]
        bboxes = [np.array(bbox) / ([w * 1.0, h * 1.0] * 4) for bbox in bboxes]
        bboxes = np.array(bboxes)
        data1 = self.__generate_data(copy.deepcopy(img), copy.deepcopy(bboxes), words.copy(), img_path)
        data2 = self.__generate_data(copy.deepcopy(img), copy.deepcopy(bboxes), words.copy(), img_path)

        return [data1, data2]

    def __generate_data(self, img, bboxes, words, img_path):

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]
        gt_words = np.full((self.max_word_num + 1, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        desc_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        for i, word in enumerate(words):
            desc_mask[i + 1] = 1  # desc_mask无需关心看不清的文字
            if word == '###':
                continue
            if word == '???':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if not self.use_ctc:
                if len(word) > self.max_word_len - 1:
                    gt_word[-1] = self.char2id['EOS']
                else:
                    gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:  # 这里放缩到短边为736左右
            img = random_scale(img, self.img_size[0], self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            if type(bboxes) == list:
                for i in range(len(bboxes)):
                    bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                           (bboxes[i].shape[0] // 2, 2)).astype('int32')
            else:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)),
                                    (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                # if words[i] == '###':
                #     cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            # if self.for_desc and random.random() < 1.0:
            #     img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
            if self.for_desc and random.random() < 0.5:
                # print('typing...', img_path)
                labels_uniqs = np.unique(imgs[1])
                gt_instance_before_crop = imgs[1].copy()
                imgs = random_crop_padding_4typing(imgs, self.img_size) 
                img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
                word_mask = update_word_mask(labels_uniqs, gt_instance, gt_instance_before_crop, word_mask, mask_iou=0.9)
                desc_mask = update_word_mask(labels_uniqs, gt_instance, gt_instance_before_crop, desc_mask, mask_iou=0.7)

            else:
                # print('No tping...', img_path)
                labels_uniqs = np.unique(imgs[1])
                imgs = random_rotate(imgs)
                gt_instance_before_crop = imgs[1].copy() # 如果都转出去了
                imgs = random_crop_padding(imgs, self.img_size)     # 在图片中切割, padding边界像素，输出736*736，切长边, 0.825的概率必切文字区域，
                img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
                # 识别+desc 不训练，检测是训练的
                word_mask = update_word_mask(labels_uniqs, gt_instance, gt_instance_before_crop, word_mask, mask_iou=0.9)
                desc_mask = update_word_mask(labels_uniqs, gt_instance, gt_instance_before_crop, desc_mask, mask_iou=0.7)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.is_transform and not self.for_desc: # 色彩是一个非常重要的信息，不希望
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)


        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()
        desc_mask = torch.from_numpy(desc_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,  # 水平矩形
        )
        if self.for_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask,
                desc_masks=desc_mask,
                identifies=img_path
            ))
        return data

    def load_ic15_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15_video(img, gt_path)
        return img, bboxes, words, img_path

#     def load_synthtext_single(self, img_path, gt):
#         img = get_img(img_path, self.read_type)
#         bboxes,words = gt
# #         bboxes, words = get_ann_ic15(img, gt_path)
#         return img, bboxes, words, img_path

#     def load_mtwi_single(self, img_path, gt_path):
#         img = get_img(img_path, self.read_type)
#         bboxes, words = get_ann_mtwi(img, gt_path)
#         return img, bboxes, words, img_path

#     def load_kwai_single(self, img_path, gt_path):
#         img = get_img(img_path, self.read_type)
#         bboxes, words = get_ann_ic15(img, gt_path)
#         return img, bboxes, words, img_path

#     def load_kwai_det_single(self, img_path, gt_path):
#         img = get_img(img_path, self.read_type)
#         bboxes, words = get_ann_ic15(img, gt_path)
#         return img, bboxes, words, img_path

#     def load_width_img_single(self, img_path, gt_path):
#         img = get_img(img_path, self.read_type)
#         bboxes, words = get_ann_ic15(img, gt_path)
#         return img, bboxes, words, img_path

#     def load_long_text_single(self, img_path, gt_path):
#         img = get_img(img_path, self.read_type)
#         bboxes, words = get_ann_ic15(img, gt_path)
#         return img, bboxes, words, img_path
    
    @staticmethod
    def collate_fn(batch):
        data = []
        for item in batch:
            data += [ii for ii in item]
        target = {}
        for k in data[0].keys():
            if k == 'identifies':
                img_paths = [item[k] for item in data] # 把batch组到一起
                identifies = torch.zeros((len(img_paths), 1))
                indx_list = list(set(img_paths))
                for i in range(identifies.shape[0]):
                   identifies[i][0] = indx_list.index(img_paths[i])
                target[k] = identifies
            else:
                target[k] = torch.stack([item[k] for item in data]) # 把batch组到一起
        return target


def get_ann_ic15_video(img, lines, vis=False):
#     h, w = img.shape[0:2]
    frames = img.copy()
    bboxes = []
    words = []
    for line in lines:
        word = line[8]
        
        if len(word) == 0 or word[0] == '#':
            words.append('###')
        else:
            words.append(word)
        bbox = [int(float(line[i])) for i in range(8)]
        
        points = np.array(bbox)
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        bbox = cv2.boxPoints(points).reshape((-1))
        
        if vis:
            cv2.putText(frames, word, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 2)
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]],[bbox[6], bbox[7]]], np.int32)
            cv2.polylines(frames, [points], True, (0,0,255), thickness=5)
            
        
#         bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    if vis:
        import random
        frame_id = random.randint(0,1000)
        cv2.imwrite("./outputs/pan_pp_r18_ICDAR15/show/{}.jpg".format(frame_id),frames)
        
    return np.array(bboxes), words

if __name__=='__main__':
    import sys
    sys.path.append('/share/lizhuang05/code/pan_pp.pytorch_dev')
    from utils import visulization

    data_loader = PAN_PP_COCOText(
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.5,
        read_type='cv2',
        with_rec=True,
        with_desc=True,
        direction_aug=False,
        direction_ratio=1.0,
    )
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        pin_memory=True
    )
    i = 0
    for data_batch in train_loader:
        i += 1 
        if i != 2:
            continue
        print(data_batch[0]['identifies'])
        print('-' * 20)
        for k, v in data_batch[0].items():
            if isinstance(v, list):
                print(f'k: {k}, v.shape: {len(v)}')    
            else:
                print(f'k: {k}, v.shape: {v.shape}')
        # data1 = data[1]
        # data2 = data[2]
        img_id = osp.basename(data_batch[0]['identifies'][0]).split('.')[0]
        print('绘制增强图像', osp.join('/share/lizhuang05/tmp/',  img_id+'_heamap'))
        for i, data in enumerate(data_batch):
            visulization.visual_feature(
                        out_path=osp.join('/share/lizhuang05/tmp/', img_id+'_heamap{}.jpg'.format(i)),
                        id2char=data_loader.id2char,
                        instance=data['gt_instances'],
                        word_masks=data['desc_masks'],
                        bboxes=data['gt_bboxes'], 
                        identifies=data['identifies'],
                        gt_words=data['gt_words'],
                        imgs=data['imgs'], 
                        gt_texts=data['gt_texts'], 
                        gt_kernels=data['gt_kernels'], 
                        training_masks=data['training_masks']
            )
        raise
