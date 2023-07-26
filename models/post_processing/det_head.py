import queue

import cv2
import kprocess as kp
import numpy as np
import torch
from torch.nn import functional as F
import time

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

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


def _pa(region, kernel, kernel_label_num, min_area=0):
    pred = np.zeros((kernel.shape[0], kernel.shape[1]), dtype=np.int32)
    q = queue.Queue()
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    points = np.array(np.where(kernel > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        q.put((tmpx, tmpy))
        pred[tmpx, tmpy] = kernel[tmpx, tmpy]

    q_A = queue.Queue()
    while True:
        while not q.empty():
            cur = q.get()
            cur_label = pred[cur[0], cur[1]]

            for j in range(4):
                tmpx = cur[0] + dx[j]
                tmpy = cur[1] + dy[j]
                if tmpx < 0 or tmpx >= kernel.shape[0] or tmpy < 0 or tmpy >= kernel.shape[1]:  # out bound
                    continue
                if region[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:  # text background or kernel
                    continue
                q_A.put((tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
        if q_A.empty():
            break
        else:
            # print(f"q_A size {len(q_A.queue)}")
            q_A, q = q, q_A

    return pred


def pa(region_and_kernel, min_area=0):
    """
    kernels: [2, 320, 184]
    emb: [4, 320, 184]
    """
    region_label_num, region_label = cv2.connectedComponents(region_and_kernel[0], connectivity=4)
    kernel_label_num, kernel_label = cv2.connectedComponents(region_and_kernel[1], connectivity=4)
    r = _pa(region_label, kernel_label, kernel_label_num, min_area)
    return r


def gpu_pa(region_and_kernel):
    region = region_and_kernel[0]
    kernel = region_and_kernel[1]
    # g_kernel_label = kp.connectedComponents(kernel, connectivity=4)
    # g_region_label = kp.connectedComponents(region, connectivity=4)
    g_kernel_label = torch.from_numpy(cv2.connectedComponents(kernel.cpu().numpy(), connectivity=4)[1]).cuda()
    g_region_label = torch.from_numpy(cv2.connectedComponents(region.cpu().numpy(), connectivity=4)[1]).cuda()
    expand_kernel = kp.pixel_aggregation(g_kernel_label, g_region_label)
    return expand_kernel

def get_det_results(out, img_meta, cfg):
    torch.cuda.synchronize()
    det_start = time.time()
    results = {}
    score = torch.sigmoid(out[:, 0, :, :])
    kernels = out[:, :2, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    labels = []
    kernels = kernels.to(torch.uint8)
    for i in range(kernels.shape[0]):
        label = gpu_pa(kernels[i])
        labels.append(label)
    labels = torch.stack(labels)

    torch.cuda.synchronize()
    pa_end = time.time()

    org_img_size = img_meta['org_img_size']
    img_size = img_meta['img_size'][0]

    label_nums = [torch.max(label).cpu().item() + 1 for label in labels]
    scale = []
    for i in range(len(org_img_size)):
        scale_ = (float(org_img_size[i][1]) / float(img_size[1]),
                float(org_img_size[i][0]) / float(img_size[0]))
        scale.append(scale_)


    labels_cuda = F.interpolate(labels.to(torch.float32).unsqueeze(0), (img_size[0], img_size[1]), mode='bilinear')[0]
    score_cuda = F.interpolate(score.unsqueeze(0), (img_size[0], img_size[1]), mode='bilinear')[0]
    bboxes_out = np.zeros((labels_cuda.shape[0], max(label_nums), 4), dtype=np.int32)
    instances = []

    torch.cuda.synchronize()
    score_end = time.time()

    bboxes, scores, areas = [], [], []
    loop_cnt = 0
    for i in range(labels_cuda.shape[0]):
        bboxes_tmp = []
        scores_tmp = []
        areas_tmp = []
        instances_tmp = []
        for j in range(1, label_nums[i]):
            torch.cuda.synchronize()
            t0 = time.time()
            ind = labels_cuda[i] == j
            points = torch.stack(torch.where(ind)).transpose(1, 0).cpu().numpy()
            if points.shape[0] < cfg.test_cfg.min_area:
                labels_cuda[i][ind] = 0
                continue
            torch.cuda.synchronize()
            t1 = time.time()
            score_i = torch.mean(score_cuda[i][ind]).cpu().item()
            if score_i < cfg.test_cfg.min_score:
                labels_cuda[i][ind] = 0
                continue
            areas_tmp.append(points.shape[0])

            torch.cuda.synchronize()
            t2 = time.time()
            tl = np.min(points, axis=0)  # axis=0; 每列的最小值
            br = np.max(points, axis=0) + 1
            bboxes_out[i, j] = (tl[0], tl[1], br[0], br[1])  # x1 x2 y1 y2
            instances_tmp.append(j)

            torch.cuda.synchronize()
            t3 = time.time()
            bbox = get_mini_boxes(points[:, ::-1])
            bbox = bbox * scale[i]  # 这里得到旋转矩形的四点坐标

            torch.cuda.synchronize()
            t4 = time.time()
            bbox = bbox.astype('int32')
            bboxes_tmp.append(bbox.reshape(-1))
            scores_tmp.append(score_i)
            print(f"point cost {t1 - t0}, area cost {t2 - t1}, instance cost {t3 - t2}, mini_box cost {t4 - t3}")
            loop_cnt += 1
        instances.append(instances_tmp)
        bboxes.append(bboxes_tmp)
        scores.append(scores_tmp)
        areas.append(areas_tmp)

    torch.cuda.synchronize()
    loop_end = time.time()

    print(f"pa cost {pa_end - det_start}, get scores cost {score_end - pa_end}, loop cost {loop_end - score_end}, loop count {loop_cnt}")
    results['bboxes'] = bboxes  # 相对于原始图片的size
    results['scores'] = scores
    results['areas'] = areas

    results['label'] = labels_cuda
    results['bboxes_out'] = bboxes_out  # 存储左上点 右下点坐标
    results['instances'] = [torch.from_numpy(np.array(i)).cuda() for i in instances] # 实例标记
    return results
