import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import visulization
from .post_processing.det_postprocess import DetPost
from pytorch_metric_learning import losses

class VisualEncoder(nn.Module):
    def __init__(self, vector_dim):
        super(VisualEncoder, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Conv2d(128, vector_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(vector_dim, vector_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool2d((1, 1), return_indices=False)
                    )

    def forward(self, x):
        return self.encoder(x)
    
class PAN_CL(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head,
                 recognition_head=None,
                 description_head=None,
                use_mulloss=False):
        super(PAN_CL, self).__init__()
        self.backbone = build_backbone(backbone)

        in_channels = neck.in_channels
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        
        self.visual_encoder = VisualEncoder(128)
        
        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.rec_head = None
        if recognition_head:
            self.rec_head = build_head(recognition_head)
#         if description_head:
#             self.desc_head = build_head(description_head)
        self.s_det =  nn.Parameter(-1.05 * torch.ones(1))
        self.s_rec =  nn.Parameter(-1.05 * torch.ones(1))
        self.s_desc =  nn.Parameter(-1.05 * torch.ones(1))
        self.use_mulloss = use_mulloss
        self.det_postprocess = DetPost(cpu_threads=4)
        self.loss_func = losses.NTXentLoss(temperature=0.1)   # Follow SimCLR
        
    # 自监督loss
    def loss(self, embeddings, identifies, reduce=True):
        """
        @description  :
        ---------
        @param  :  embeddings: (N 384)
        -------
        @Returns  :
        -------
        """
        # print(f"embeddings数量: {embeddings.shape[0]}", f"identifies数: {identifies.shape[0]}")
        # print(f"embeddings device: {embeddings.device}", f"identifies device: {identifies.device}")
        # 使用identifies生成label
        idx4pair = torch.zeros((identifies.shape[0]), device=identifies.device, dtype=torch.int)
        label_map = {}
        idx = 0
        for i in range(identifies.shape[0]):
            origin_id = tuple(identifies[i].cpu().numpy().tolist())
            if origin_id not in label_map:
                label_map[origin_id] = idx
                idx += 1
            idx4pair[i] = label_map[origin_id]

        # print(identifies)
        # if idx4pair.shape[0] <= 1 or len(label_map) == idx4pair.shape[0]:
        #     # print('不能计算loss_desc')
        #     return {'loss_desc': idx4pair.new_full((1,), 0, dtype=torch.float32)}
        # print(idx4pair)
        loss_desc = self.loss_func(embeddings, idx4pair)
        loss_desc = loss_desc.view(-1)
        losses = {'loss_desc': loss_desc}
        return losses

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                desc_masks=None,
                img_metas=None,
                identifies=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs) # 2, 64, 184, 184

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel 这里将通道数全部变成128
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        # FFM
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))
            start = time.time()

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            if not cfg.train_cfg.freeze_det:  # 冻结检测分支时，不进行loss 计算加速训练
                loss_det = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes, )
            else:
                loss_det = {
                        'loss_text': f.new_full((1,), 0, dtype=torch.float32),
                        'loss_kernels': f.new_full((1,), 0, dtype=torch.float32),
                        'loss_emb': f.new_full((1,), 0, dtype=torch.float32),
                        'iou_text': f.new_full((1,), 0, dtype=torch.float32),
                        'iou_kernel': f.new_full((1,), 0, dtype=torch.float32)
                }
            outputs.update(loss_det)
        else:
#             start = time.time()
#             det_out = self._upsample(det_out, imgs.size(), cfg.test_cfg.scale)
#             det_res = self.det_head.get_results(det_out, img_metas, cfg)
            
#             print("before_process:",time.time()-start)
#             start = time.time()
            
#             det_res = self.det_head.get_results(det_out, img_metas, cfg)
            det_res = self.det_postprocess.get_det_result(det_out, img_metas, cfg)
            
#             print("post_process:",time.time()-start)
#             start = time.time()

            outputs.update(det_res)
        
        if self.rec_head is not None:
            if self.training:
                # 如果识别分支和desc分支都不训练，直接x_crop 为None就可以了
#                 print(cfg.train_cfg.freeze_rec)
#                 print(cfg.train_cfg.freeze_desc)
                if cfg.train_cfg.freeze_rec and cfg.train_cfg.freeze_desc:
                    x_crops = None
                else:
                    x_crops, words, tlbrs, identify_labels, rec_masks = self.rec_head.extract_feature(
                                f, (imgs.size(2), imgs.size(3)), 
                                gt_instances * training_masks, 
                                gt_bboxes, 
                                gt_words=gt_words, 
                                word_masks=word_masks,
                                desc_masks=desc_masks,
                                identifies=identifies)
                
                if x_crops is not None: # N * 128 * 8 * 32；  x_crops存储的就是ROI mask之后的特征
#                     out_rec = self.rec_head(x_crops, words) # N * 32 (最大字符串长度) * voc_size 这是识别的信息 
#                     if not cfg.train_cfg.freeze_rec:
#                         loss_rec = self.rec_head.loss(out_rec, words, rec_masks, reduce=False)
#                     else:
                    
                    # TODO
                    if not cfg.train_cfg.freeze_desc:
                        # 训练的 bbox使用gt 
                        multi_info_feature = self.visual_encoder(x_crops).squeeze(-1).squeeze(-1)
#                         multi_info_feature = self.desc_head(x_crops, tlbrs, out_rec) # N 384
                        loss_desc = self.loss(multi_info_feature, identify_labels)
                    else:
                        loss_desc = {
                            'loss_desc': f.new_full((1,), 0, dtype=torch.float32)
                        }
                else:
                    loss_desc = {
                        'loss_desc': f.new_full((1,), 0, dtype=torch.float32)
                        }
                   
                
                loss_rec = {
                        'loss_rec': f.new_full((1,), 0, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), 0, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
                outputs.update(loss_desc)
                l_det = torch.mean(loss_det['loss_kernels']) + torch.mean(loss_det['loss_emb']) + torch.mean(loss_det['loss_text'])
                l_desc = torch.mean(loss_desc['loss_desc'])
#                 l_rec = loss_rec['loss_rec']
                
#                 l_rec = torch.mean(l_rec)

#                 if self.use_mulloss == True:
#                     # print("多任务学习损失")
#                     loss_sum = torch.exp(-self.s_det) * l_det + torch.exp(-self.s_rec) * l_rec + torch.exp(-self.s_desc) * l_desc + \
#                             (self.s_det + self.s_rec + self.s_desc)
#                 else:
                    # print("直接和")
                loss_sum = l_det + l_desc

                outputs.update({'loss': loss_sum})

            
            # 推理
            else:
                start = time.time()
                if len(det_res['bboxes']) > 0:
                    x_crops, _, tlbrs, _, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(det_res['label'], dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(det_res['bboxes_h'], dtype=torch.long),
                        unique_labels=det_res['instances'])
                    
                    if cfg.report_speed:
                        torch.cuda.synchronize()
                        outputs.update(dict(
                            rec_align_time=time.time() - start
                        ))
                        start = time.time()

#                     words, word_scores, out_rec = self.rec_head.forward(x_crops)
#                     if cfg.report_speed:
#                         torch.cuda.synchronize()
#                         outputs.update(dict(
#                             rec_time=time.time() - start
#                         ))
#                         start = time.time()

                    # print(out_rec.shape)
                    # desc 预测
#                     multi_info_feature = self.desc_head(x_crops, tlbrs, out_rec) # N 384
                    multi_info_feature = self.visual_encoder(x_crops).squeeze(-1).squeeze(-1)
                    multi_info_feature = multi_info_feature.cpu().numpy()
                    # print(multi_info_feature.shape)
                    if cfg.report_speed:
                        torch.cuda.synchronize()
                        outputs.update(dict(
                            desc_time=time.time() - start
                        ))
                        start = time.time()

                else:
#                     words = []
#                     word_scores = []
                    multi_info_feature = None

                outputs.update(dict(
#                     words=words,
#                     word_scores=word_scores,
                    label='',
                    multi_info_feature=multi_info_feature
                ))

        return outputs
