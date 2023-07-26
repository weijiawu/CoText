#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visulization.py
@Time    :   2021/07/30 12:17:07
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
from  matplotlib import pyplot as plt
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
import copy
import torch.nn.functional as F
import matplotlib.image as mpimg # mpimg 用于读取图片



def paint_chinese_opencv(im, chinese, pos, color, font_size=35):
    img_PIL = Image.fromarray(im)
    font = ImageFont.truetype('/share/lizhuang05/tmp/simhei.ttf', font_size)
    fillColor = color  #  (255,0,0)
    position = pos #  (100,100)
    # if not isinstance(chinese, unicode):
    #     chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL, mode='RGB')
    draw.text(position, chinese, font=font, fill=fillColor)
    # img = cv2.cvtColor(, cv2.COLOR_RGB2BGR)
    return np.asarray(img_PIL)

def plot_bboxs(image_path, bboxes, contents, color=(0, 255, 0)):
    image = cv2.imread(image_path)
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(3, max(image.shape[1], image.shape[0]) / 1000.)
    line_thickness = 3 * int(max(image.shape[1], image.shape[0]) / 400.)
    for i, bbox in enumerate(bboxes):
        content = contents[i]
        if content is None:
            content = "###"
        x1, y1,x2, y2, x3, y3, x4, y4 = bbox[:8]
        intbox = list(map(lambda x: int(float(x)), (x1, y1, x2, y2, x3, y3, x4, y4)))
        intbox_np = np.array(intbox).reshape(-1,1,2)
        id_pos = (intbox[0], intbox[1])
        # 画四边形
        cv2.polylines(im, [intbox_np], True, color=color, thickness=line_thickness)
        im = paint_chinese_opencv(im, content, id_pos, color, font_size=text_scale*10)
    return im

def visual_feature(out_path,
                    id2char,
                    instance,
                    word_masks,
                    bboxes, 
                    identifies,
                    gt_words,
                    imgs, 
                    gt_texts, 
                    gt_kernels, 
                    training_masks
                    ):
    
    plt.figure(figsize=(10, 6 * 6))
    img = imgs[0].permute(1,2,0).data.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = (img * std + mean).numpy()
    img*=250.0
    img = img.astype(np.uint8)
    img_plot = np.copy(img)
    gt_text = gt_texts[0]
    gt_kernel = gt_kernels[0].squeeze(0)
    training_mask = training_masks[0]

    instance_ = instance[0:1]
    unique_labels_, _ = torch.unique(instance_, sorted=True, return_inverse=True)
    word_masks_ = word_masks[0]
    bboxes_ = bboxes[0]
    identifie = identifies[0]
    gt_words_ = gt_words[0]
    
    print(unique_labels_)
#     print(bboxes_)
    for label in unique_labels_:
        if label == 0:
            continue
        if word_masks is not None and word_masks_[label] == 0:
            continue
        t, l, b, r = bboxes_[label]
        print(t, l, b, r)
        mask = (instance_[:, t:b, l:r] == label).float()
        mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=(3, 3), stride=1, padding=1)[0]
        if torch.sum(mask) == 0:
            print("continue")
            continue
        # 画label就可以了          
        gt_word = ''
        for j, char_id in enumerate(gt_words_[label]):
            char_id = int(char_id)
            if char_id == 4711:
                break
            if id2char[char_id] in ['PAD', 'UNK']:
                continue
            gt_word += id2char[char_id]
        # 开始画图形
        img_plot = paint_chinese_opencv(img_plot, str(gt_word), 
                                        (r.item(), t.item()), 
                                        (0,255,0))
        
        img_plot = paint_chinese_opencv(img_plot, str(label.item()), 
                                        (l.item(), t.item()), 
                                        (0,255,0))
        img_plot = cv2.rectangle(img_plot, (l.item(), t.item()),  (r.item(), b.item()), (255, 0, 0))

    ori_img = mpimg.imread(identifie)
    identifie = osp.basename(identifie)
    plt.subplot(6, 1, 1)
    plt.imshow(ori_img)
    plt.colorbar()
    plt.title('Original img: {}'.format(identifie), fontsize=15)

    # 先画原始图片
    plt.subplot(6, 1, 2)
    plt.imshow(img)
    plt.colorbar()
    plt.title('Aug img: {}'.format(identifie), fontsize=15)

    # # 识别
    plt.subplot(6, 1, 3)
    plt.imshow(gt_text.data.cpu().numpy().squeeze())
    plt.colorbar()
    plt.title('text region: {}'.format(identifie), fontsize=15)

    plt.subplot(6, 1, 4)
    plt.imshow(gt_kernel.data.cpu().numpy().squeeze())
    plt.colorbar()
    plt.title('kernel: {}'.format(identifie), fontsize=15)

    plt.subplot(6, 1, 5)
    plt.imshow(training_mask.data.cpu().numpy().squeeze())
    plt.colorbar()
    plt.title('training_mask: {}'.format(identifie), fontsize=15)
    
    plt.subplot(6, 1, 6)
    plt.imshow(img_plot)
    plt.colorbar()
    plt.title('img_instance: {}'.format(identifie), fontsize=15)

    plt.savefig(out_path)
    plt.clf()

        


