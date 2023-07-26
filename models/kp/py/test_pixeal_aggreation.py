import queue
import time

import cv2
import numpy as np
import torch
import kprocess as kp


def _pa(region, kernel):
    pred = np.zeros((kernel.shape[0], kernel.shape[1]), dtype=np.int32)

    # for i in range(1, kernel_label_num):
    #     ind = kernel == i
    #     area = np.sum(ind)
    #     if area < min_area:
    #         kernel[ind] = 0

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


if __name__ == '__main__':
    region_and_kernel = np.load("region_and_kernel.npy")
    region_label_num, region_label = cv2.connectedComponents(region_and_kernel[0], connectivity=4)
    kernel_label_num, kernel_label = cv2.connectedComponents(region_and_kernel[1], connectivity=4)
    g_region_label = torch.from_numpy(region_label).cuda()
    g_kernel_label = torch.from_numpy(kernel_label).cuda()

    r = _pa(region_label, kernel_label)
    g_r = kp.pixel_aggregation(g_kernel_label, g_region_label).cpu().numpy()
    diff = np.abs(r - g_r)
    print(f"diff max {np.max(diff)}")

    # region_label_num, region_label = cv2.connectedComponents(region_and_kernel[1], connectivity=4)
    # region_label = torch.from_numpy(region_label)
    # g_region_label = kp.connectedComponents(torch.from_numpy(region_and_kernel[1]).cuda(), connectivity=4).cpu()
    #
    # def show_value(a, b, i):
    #     print(a[a==i])
    #     print(b[b==i])
    #
    # def show_count(a, b):
    #     a_uniq = torch.unique(a)
    #     b_uniq = torch.unique(b)
    #     for i in a_uniq:
    #         print(f"{i} : {torch.sum(a == i)}")
    #     print('-' * 80)
    #     for i in b_uniq:
    #         print(f"{i} : {torch.sum(b == i)}")
    #
    # show_count(region_label, g_region_label)
