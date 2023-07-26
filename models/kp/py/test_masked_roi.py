import torch
import torch.nn.functional as F
import numpy as np
import os
import kprocess as kp


def masked_roi(image, bboxes_out, kernel, unique_labels, feature_size):
    x_crops = []
    np_unique_labels = unique_labels.numpy()
    for idx, label_id in enumerate(np_unique_labels):
        t, l, b, r = bboxes_out[idx]
        mask = (kernel[t:b + 1, l:r + 1] == label_id).float()
        # # 让 mask 区域往外扩充 1 个像素
        # mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=(3, 3), stride=1, padding=1)[0]

        x_crop = image[:, t:b + 1, l:r + 1] * mask  # 这里就是截取 经过mask后的文字对象的图像特征
        _, h, w = x_crop.size()
        if h > w * 1.5:  # 判定是否为竖排文字
            x_crop = x_crop.transpose(1, 2)
        x_crop = F.interpolate(x_crop.unsqueeze(0), feature_size, mode='bilinear', align_corners=True)
        x_crops.append(x_crop)  # 直接将图像特征双线性差值到固定大小，也就是 8*32，
    cat_x_crops = torch.cat(x_crops, 0)
    return cat_x_crops


if __name__ == '__main__':
    data_path = '/Users/liupeng/code/pan_pp'
    image = torch.from_numpy(np.load(os.path.join(data_path, "image.npy")))
    bboxes = torch.from_numpy(np.load(os.path.join(data_path, "bbox.npy")))
    kernel = torch.from_numpy(np.load(os.path.join(data_path, "kernel.npy")))
    label = torch.from_numpy(np.load(os.path.join(data_path, "label.npy")))
    result_crops = torch.from_numpy(np.load(os.path.join(data_path, "result_crops.npy")))
    bboxes = bboxes[label]

    # c = 1
    # h = 16
    # w = 16
    # num_boxes = 2
    #
    # image = torch.tensor([
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 2, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 4, 5, 6, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 7, 8, 9, 0, 0, 1, 1, 1, 1, 0, 0, 19, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 10, 11, 12,
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 13, 14, 15,
    #     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 18,
    #     0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #     0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    #     0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([c, h, w]).float()
    # kernel = torch.tensor([
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 2, 2, 2, 0, 0, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 3, 3,
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3,
    #     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3,
    #     0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #     0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #     0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    #     0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    #     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # ]).reshape([h, w]).int()
    #
    # bboxes = torch.tensor([1, 1, 3, 3, 3, 12, 6, 15]).reshape([num_boxes, 4]).int()
    # label = torch.tensor([2, 3]).int()

    bboxes_cuda = bboxes.int().cuda()
    image_cuda = image.float().cuda()
    kernel_cuda = kernel.int().cuda()
    label_cuda = label.int().cuda()

    feature_size = [8, 32]
    this_crops = masked_roi(image, bboxes, kernel, label, feature_size)
    crop_h, crop_w = feature_size
    gpu_crops = kp.masked_roi(image_cuda, bboxes_cuda, kernel_cuda, label_cuda, crop_h, crop_w)
    diff = torch.abs(this_crops - gpu_crops.cpu())

    assert diff.max().item() > 1e-5, 'has diff'
