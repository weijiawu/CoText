#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Date    : 2021/09/29 15:22:13
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

torch.manual_seed(23)
torch.cuda.manual_seed(23)
np.random.seed(23)
random.seed(22)


ic15_root_dir = '/share/wuweijia/Data/ICDAR2015/'
ic15_train_data_dir = ic15_root_dir + 'train_image/'
ic15_train_gt_dir = ic15_root_dir + 'train_gt/'

coco_root_dir = '/share/wuweijia/Data/COCOTextV2/'
coco_train_data_dir = coco_root_dir + 'train_image/'
coco_train_gt_dir = coco_root_dir + 'train_gt_icd15/'


synth_root_dir = '/share/wuweijia/Data/SynthText/'
synth_train_data_dir = synth_root_dir
synth_train_gt_dir = synth_root_dir + 'gt.mat'


ic15_video_train_data_dir = '/share/lizhuang05/datasets/ICDAR2015_VIDEO/train_images' #home/wangjue_Cloud
ic15_video_train_gt_dir = '/share/lizhuang05/datasets/ICDAR2015_VIDEO/train_gts'   # home/wangjue_Cloud

class PAN_PP_COCOText(data.Dataset):
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

        # ic15
        self.img_paths['ic15'] = []
        self.gts['ic15'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = ic15_train_data_dir + img_name
            self.img_paths['ic15'].append(img_path)

            gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            # gt_name = img_name.split('.')[0] + '.txt'
            gt_path = ic15_train_gt_dir + gt_name
            self.gts['ic15'].append(gt_path)
            
        self.img_paths['ic15'] = self.img_paths['ic15'] * 100
        self.gts['ic15'] = self.gts['ic15'] * 100
        
        self.img_num += len(self.img_paths['ic15'])
        
        # cocotext
        self.img_paths['cocotext'] = []
        self.gts['cocotext'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(coco_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(coco_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = coco_train_data_dir + img_name
            
            # gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_name = img_name.split('.')[0] + '.txt'
            gt_path = coco_train_gt_dir + gt_name
            if not osp.exists(gt_path):
                pass
            else:
                self.img_paths['cocotext'].append(img_path)
                self.gts['cocotext'].append(gt_path)
                
        self.img_paths['cocotext'] = self.img_paths['cocotext'] * 20
        self.gts['cocotext'] = self.gts['cocotext'] * 20
        
        self.img_num += len(self.img_paths['cocotext'])
        
        
        # synthtext
        self.img_paths['synthtext'] = []
        self.gts['synthtext'] = []
        data = scio.loadmat(synth_train_gt_dir)
        img_paths = data['imnames'][0]
        gts = data['wordBB'][0]
        texts = data['txt'][0]
        for i in range(len(img_paths)):
            image_path = img_paths[i][0]
            roo_image_path = os.path.join(synth_train_data_dir,image_path)
            
            bboxes = np.array(gts[i])
            bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
            bboxes = bboxes.transpose(2, 1, 0)
            bboxes = np.reshape(
                bboxes, (bboxes.shape[0], -1))
            
            words = []
            for idx,text in enumerate(texts[i]):
                
                text = text.replace('\n', ' ').replace('\r', ' ')
                words.extend([w for w in text.split(' ') if len(w) > 0])
            
            self.img_paths['synthtext'].append(roo_image_path)
            self.gts['synthtext'].append((bboxes,words))
        self.img_num += len(self.img_paths['synthtext'])
        
        
        # ic15_video
        self.img_paths['ic15_video'] = []
        self.gts['ic15_video'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(ic15_video_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(ic15_video_train_data_dir, '.png')])
#         img_names = ["{}.jpg".format(i+1) for i in range(len(img_names))]
#         print(img_names)
        for idx, img_name in enumerate(img_names):
            img_path = osp.join(ic15_video_train_data_dir, img_name)
            self.img_paths['ic15_video'].append(img_path)

            gt_name = img_name.split('.')[0] + '.txt'
            gt_path = osp.join(ic15_video_train_gt_dir, gt_name)
            self.gts['ic15_video'].append(gt_path)
            
        self.img_paths['ic15_video'] = self.img_paths['ic15_video'] * 40
        self.gts['ic15_video'] = self.gts['ic15_video'] * 40
        self.img_num += len(self.img_paths['ic15_video'])
        
        
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

    # For 自监督
    def __getitem__(self, index): 

        if index == 0:        # 这里需要进行顺序打乱
            print('shuffle traing data!')
            random.shuffle(self.data_list)
        dataset_cat, img_path, gt_path = self.data_list[index]
        if dataset_cat == 'ic15':
            img, bboxes, words, img_path = self.load_ic15_single(img_path, gt_path)
        elif dataset_cat == 'cocotext':
            img, bboxes, words, img_path = self.load_ic15_single(img_path, gt_path)
        elif dataset_cat == "synthtext":
            img, bboxes, words, img_path = self.load_synthtext_single(img_path, gt_path)
        elif dataset_cat == "ic15_video":
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
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_synthtext_single(self, img_path, gt):
        img = get_img(img_path, self.read_type)
        bboxes,words = gt
#         bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_mtwi_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_mtwi(img, gt_path)
        return img, bboxes, words, img_path

    def load_kwai_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_kwai_det_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_width_img_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_long_text_single(self, img_path, gt_path):
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path
    
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
