#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pan_pp_des_head.py
@Time    :   2021/07/27 18:41:05
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   panpp的 多信息融合分支head
'''

# here put the import lib
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from ..loss import build_loss
from ..loss import pytorch_loss as losses
# from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

class CoText_TrackHead(nn.Module):
    def __init__(self, 
                    vector_dim,
                    img_size,
                    voc,
                    loss_weight=0.5):
        super(CoText_TrackHead, self).__init__()

        self.visual_encoder = VisualEncoder(vector_dim)
        self.position_encoder = PositionEncoder(img_size)
        self.sequence_encoder = SequenceEncoder(voc, vector_dim=vector_dim)

#         self.loss_func = losses.TripletMarginLoss(margin=0.5)
        self.loss_func = losses.NTXentLoss(temperature=0.1)   # Follow SimCLR
#         self.loss_func = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=CosineSimilarity())
        
        self.loss_weight = loss_weight
        self.fuse_conv = nn.Sequential(
                            nn.Conv2d(int(vector_dim * 3), int(vector_dim * 3), kernel_size=1),
                            # nn.BatchNorm2d(int(vector_dim * 3)),
                            nn.ReLU(inplace=True)
        ) 
        self.conv = nn.Conv2d(int(vector_dim * 3), vector_dim, kernel_size=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #              [N, 128, 8 , 32], [N, 4, 128, 1], [N, 32, 4714]
    def forward(self, x_visual, tlbrs, x_seq):
        f_visual = self.visual_encoder(x_visual) # N 128 1 1
        # print(f_visual)
        f_pos = self.position_encoder(tlbrs) # N 128 1 1
        # print(f_pos)
        f_seq = self.sequence_encoder(x_seq) # N 128 1 1
#         print(f_seq)
        multi_info_feature = torch.cat([f_visual, f_pos, f_seq], dim=1)
#         multi_info_feature = torch.cat([f_visual, f_visual, f_visual], dim=1)
        f_fuse = self.fuse_conv(multi_info_feature)
        f_fuse_1 = self.conv(f_fuse)
        
        out_feature = f_fuse_1.squeeze(-1).squeeze(-1) # N 384 1 1 

        return out_feature  # N 384
        # return multi_info_feature.squeeze(-1).squeeze(-1)

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

        
        loss_desc = self.loss_func(embeddings, idx4pair) * self.loss_weight # 只有一对儿相同类别，loss返回0
        loss_desc = loss_desc.view(-1)
        losses = {'loss_desc': loss_desc}
        return losses


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



class PositionEncoder(nn.Module):
    def __init__(self,  
                    img_size=736,
                    vector_dim=128):

        super(PositionEncoder, self).__init__()
        self.encoder = nn.Sequential(
                    nn.Conv2d(4, 64, kernel_size=1, stride=1),
                    nn.BatchNorm2d(64),
                    # nn.Conv2d(4, vector_dim, kernel_size=1, stride=1),
                    # nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.BatchNorm2d(128),
                    # nn.Conv2d(vector_dim, vector_dim, kernel_size=1, stride=1),
                    # nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool2d((1, 1), return_indices=False)
                    )
        
        self.vector_dim = int(vector_dim // 4) * 2
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            self.height = img_size[0]
            self.width = img_size[1]
        else:
            self.height = img_size
            self.width = img_size
        self.half = False
    def forward(self, x):
        # 2, 4
#         print(x.dtype)
        x = self.coordinate_embeddings(x)  # N, 4, 128, 1
        # 再经过两个全连阶层进行编码 [N, 4]
#         print(x.dtype)
        if self.half:
            x = self.encoder(x.half())
        else:
            x = self.encoder(x)

        return x

    def coordinate_embeddings(self, boxes):
        """
        Coordinate embeddings of bounding boxes
        :param boxes: [K, 6] ([x1, y1, x2, y2, w_image, h_image])
        :param dim: sin/cos embedding dimension
        :return: [K, 4, 2 * dim]
        referenced from https://github.com/jackroos/VL-BERT/blob/master/common/utils/bbox.py#L33-L65
        """

        num_boxes = boxes.shape[0]
        # transform tlbr to (x_c, y_c, w, h) format
        boxes_ = boxes.new_zeros((num_boxes, 4))
        boxes_[:, 0] = (boxes[:, 1] + boxes[:, 3]) / 2
        boxes_[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes_[:, 2] = boxes[:, 3] - boxes[:, 1]
        boxes_[:, 3] = boxes[:, 2] - boxes[:, 0]
        boxes = boxes_
        
#         print(boxes.shape)

        # position
        pos = boxes.new_zeros((num_boxes, 4))
        pos[:, 0] = boxes[:, 0] / self.width * 100
        pos[:, 1] = boxes[:, 1] / self.height * 100
        pos[:, 2] = boxes[:, 2] / self.width * 100
        pos[:, 3] = boxes[:, 3] / self.height * 100
        
        
        # sin/cos embedding
        dim_mat = 1000 ** (torch.arange(self.vector_dim, dtype=boxes.dtype, device=boxes.device) / self.vector_dim)
#         print(dim_mat.shape)
#         print(dim_mat.view((1, 1, -1)).shape)
        sin_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).sin()
        cos_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).cos()
#         print(cos_embedding.shape)
        pos_embedding = torch.cat((sin_embedding, cos_embedding), dim=-1).unsqueeze(-1)
#         print(pos_embedding.shape)
        return pos_embedding


class SequenceEncoder(nn.Module):
    def __init__(self, voc, vector_dim):
        super(SequenceEncoder, self).__init__()
        self.vocab_size = len(voc)
        self.encoder = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=1, stride=1),
                    nn.BatchNorm2d(64),
                    # nn.Conv2d(32, vector_dim, kernel_size=1, stride=1),
                    # nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.BatchNorm2d(128),
                    # nn.Conv2d(vector_dim, vector_dim, kernel_size=1, stride=1),
                    # nn.BatchNorm2d(vector_dim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool2d((1, 1), return_indices=False)
                    )

    def forward(self, x):
        # input  [N, 32, 38]
        x = x.unsqueeze(-1)  # [N, 32, 38, 1]
        x = self.encoder(x)  # [N, 128, 1, 1]
        return x

def _get_simclr_projection_head(num_ftrs: int, out_dim: int):
    """Returns a 2-layer projection head.
    Reference (07.12.2020):
    https://github.com/google-research/simclr/blob/master/model_util.py#L141
    """
    modules = [
        nn.Linear(num_ftrs, num_ftrs),
        nn.BatchNorm1d(num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs, out_dim)
    ]
    return nn.Sequential(*modules)



if __name__ == '__main__':
    # [N, 128, 8 , 32], [N, 4, 128, 1], [N, 32, 4714]
    x1 = torch.ones((2, 128, 16 , 32))
    x2 = torch.empty((2, 4))
    x3 = torch.ones((2, 32, 4714))
    
    import sys
    sys.path.append('/share/lizhuang05/code/pan_pp.pytorch_dev')
    from dataset.pan_pp import pan_pp_jointTrain
    voc, char2id, id2char = pan_pp_jointTrain.get_vocabulary('CHINESE', use_ctc=True)
    model = PAN_PP_DescHead(vector_dim=128,
                            img_size=736,voc=voc)


    y = model.forward(x1, x2, x3)
    print(y.shape)
