import queue
import cv2
import kprocess as kp
import numpy as np
import torch
from torch.nn import functional as F
import time
from multiprocessing.pool import ThreadPool

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

def gpu_pa(region_and_kernel):
    region = region_and_kernel[0]
    kernel = region_and_kernel[1]
    # g_kernel_label = kp.connectedComponents(kernel, connectivity=4)
    # g_region_label = kp.connectedComponents(region, connectivity=4)
    g_kernel_label = torch.from_numpy(cv2.connectedComponents(kernel.cpu().numpy(), connectivity=4)[1]).cuda()
    g_region_label = torch.from_numpy(cv2.connectedComponents(region.cpu().numpy(), connectivity=4)[1]).cuda()
    expand_kernel = kp.pixel_aggregation(g_kernel_label, g_region_label)
    del g_kernel_label, g_region_label
    return expand_kernel

class DetPost():
    def __init__(self, cpu_threads=4):
        # self.cfg = cfg
        self.pool = ThreadPool(cpu_threads)
        self.bboxes_pool = ThreadPool(cpu_threads)

    def run(self, label_num, label_cuda, score_cuda, bbox_out, scale):
        bboxes_tmp, scores_tmp, areas_tmp, instances_tmp, points_list = [], [], [], [], []
        for i in range(1, label_num):
            ind = label_cuda == i
            points = torch.stack(torch.where(ind)).transpose(1, 0).cpu().numpy()
            score_i = torch.mean(score_cuda[ind]).cpu().item()

            # if points.shape[0] > self.cfg.test_cfg.min_area and score_i > self.cfg.test_cfg.min_score:
            if points.shape[0] > self.cfg.test_cfg.min_area and score_i > self.cfg.test_cfg.min_score:
                areas_tmp.append(points.shape[0])
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bbox_out[i] = (tl[0], tl[1], br[0], br[1])
                instances_tmp.append(i)
                scores_tmp.append(score_i)
                points_list.append(points)
            else:
                label_cuda[ind] = 0

        def get_bboxes_tmp(points):
            bbox = get_mini_boxes(points[:, ::-1])
            bbox = bbox * scale
            return bbox.astype('int32').reshape(-1)

        bboxes_tmp = self.bboxes_pool.map(get_bboxes_tmp, points_list)
        return instances_tmp, bboxes_tmp, scores_tmp, areas_tmp, label_cuda, bbox_out

    def get_det_result(self, out, img_meta, cfg):
        self.cfg = cfg
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
      
        org_img_size = img_meta['org_img_size']
        img_size = img_meta['img_size'][0]
        valid_size = img_meta['valid_size']
        label_nums = [torch.max(label).cpu().item() + 1 for label in labels]

        scale = []
        for i in range(len(org_img_size)):
            scale_ = (float(org_img_size[i][1]) / float(valid_size[i][1]),
                    float(org_img_size[i][0]) / float(valid_size[i][0]))
            scale.append(scale_)

        labels_cuda = F.interpolate(labels.to(torch.float32).unsqueeze(0), (img_size[0], img_size[1]), mode='nearest')[0]
        score_cuda = F.interpolate(score.unsqueeze(0), (img_size[0], img_size[1]), mode='nearest')[0]
        bboxes_out = np.zeros((labels_cuda.shape[0], max(label_nums), 4), dtype=np.int32)

        labels_cuda_list = [labels_cuda[i] for i in range(labels_cuda.shape[0])]
        score_cuda_list = [score_cuda[i] for i in range(score_cuda.shape[0])]
        bbox_out_list = [bboxes_out[i] for i in range(bboxes_out.shape[0])]

        async_reuslts = []
        for i in range(len(labels_cuda_list)):
            async_reuslts.append(self.pool.apply_async(self.run, 
                (label_nums[i], labels_cuda_list[i], score_cuda_list[i], bbox_out_list[i], scale[i])))

        out_results = [x.get() for x in async_reuslts]
        instances = [out_result[0] for out_result in out_results]
        bboxes = [out_result[1] for out_result in out_results]
        scores = [out_result[2] for out_result in out_results]
        areas = [out_result[3] for out_result in out_results]
        labels_cuda = [out_result[4] for out_result in out_results]
        bboxes_out = [out_result[5] for out_result in out_results]

        # print(bboxes)
        results['bboxes'] = bboxes[0]
        results['scores'] = scores[0]
        results['areas'] = areas[0]

        # results['label'] = torch.stack(labels_cuda)
        results['label'] = torch.stack(labels_cuda).squeeze(0).cpu().numpy()
        results['bboxes_h'] = np.stack(bboxes_out)
        # results['instances'] = [torch.from_numpy(np.array(i)).cuda() for i in instances]
        results['instances'] = instances

        return results
