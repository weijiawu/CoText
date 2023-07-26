import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import CoordConv2d
from ..loss import build_loss, ohem_batch, iou
from ..post_processing import pa
import cv2
import time


class CoText_DetHead(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel,
                 loss_emb,
                 use_coordconv=False):
        super(CoText_DetHead, self).__init__()
        if not use_coordconv:
            self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = CoordConv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        return out

    def get_results(self, out, img_meta, cfg):
        results = {}
        if cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])

        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
 
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()
        
        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)
        
        
        label = pa(kernels, emb, cfg.test_cfg.min_kernel_area / (cfg.test_cfg.scale **2))
        
        # if cfg.report_speed:
        #     torch.cuda.synchronize()
        #     results['det_pa_time'] = time.time() - start

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]
        valid_size = img_meta['valid_size'][0]

        label_num = np.max(label) + 1
        scale = (float(org_img_size[1]) / float(valid_size[1]),
                 float(org_img_size[0]) / float(valid_size[0]))
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        with_rec = hasattr(cfg.model, 'recognition_head')
        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        areas = []
        for i in range(1, label_num):   # 第0为背景
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))
            # 这里比的是原图的kernel的面积进行过滤，太保守了
            # min_points_area = img_size[0]*img_size[1]/736/1280 * cfg.test_cfg.min_area
            # if i == 1:
            #     print('min_area:', min_points_area, '\torg_img_size:', org_img_size)
            if points.shape[0] <cfg.test_cfg.min_area: #  / (cfg.test_cfg.scale **2):
                label[ind] = 0
                continue
            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue
            areas.append(points.shape[0])
            if with_rec:
                tl = np.min(points, axis=0) # axis=0; 每列的最小值
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1]) # x1 x2 y1 y2
                instances[0].append(i)

            if cfg.test_cfg.bbox_type == 'rect':
                bbox = self.get_mini_boxes(points[:, ::-1]) 
                bbox = bbox * scale       # 这里得到旋转矩形的四点坐标
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale
 
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        results['bboxes'] = bboxes    # 相对于原始图片的size
        results['scores'] = scores
        results['areas'] = areas
        
        if cfg.report_speed:
            torch.cuda.synchronize()
            results['det_post_time'] = time.time() - start

        if with_rec:
            results['label'] = label
            results['bboxes_h'] = bboxes_h    # 存储左上点 右下点坐标
            results['instances'] = instances  # 实例标记
        return results

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes):
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_masks)   # 负采样
        # loss_text = dice_loss(texts, gt_texts, selected_masks, reduce=False)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = {'loss_text': loss_text, 'iou_text': iou_text}

        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            # loss_kernel_i = dice_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts,
                         reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        loss_emb = self.emb_loss(embs,
                                 gt_instances,
                                 gt_kernels[:, -1, :, :],
                                 training_masks,
                                 gt_bboxes,
                                 reduce=False)
        losses.update(dict(
            loss_emb=loss_emb
        ))

        return losses

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

#         x, y, w, h = cv2.boundingRect(contour)
#         points = np.array([x,y, x+w,y, x+w,y+h, x,y+h]).reshape(4,2)
        
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array([points[index_1], points[index_2],
               points[index_3], points[index_4]])
        return box
