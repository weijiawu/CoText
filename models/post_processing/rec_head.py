import torch
import torch.nn.functional as F
import kprocess as kp


def masked_roi(image, bboxes_out, kernel, unique_labels, feature_size):
    x_crops = []
    np_unique_labels = unique_labels.cpu().numpy()
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


def get_rec_results(x, kernels, bboxes_out, unique_labels, feature_size=(8, 32)):
    batch_size, _, H, W = x.size()
    pad_scale = 1
    pad = x.new_tensor([-1, -1, 1, 1], dtype=torch.long) * pad_scale
    bboxes_out = bboxes_out + pad
    bboxes_out[:, :, (0, 2)] = bboxes_out[:, :, (0, 2)].clamp(0, H)
    bboxes_out[:, :, (1, 3)] = bboxes_out[:, :, (1, 3)].clamp(0, W)

    tlbrs_list = []
    x_crops_list = []
    for i in range(batch_size):
        if unique_labels[i].size(0) <= 0:
            tlbrs_list.append(None)
            x_crops_list.append(None)
            continue
        tlbrs = bboxes_out[i][unique_labels[i]]
        tlbrs_list.append(tlbrs)
        # N 128 8 32, batch中所有的roi都提取到这里了
        # x_crops = masked_roi(x[i], tlbrs, kernels[i], unique_labels[i], feature_size)
        x_crops = kp.masked_roi(x[i].float(), tlbrs.to(torch.int), kernels[i].to(torch.int), unique_labels[i].to(torch.int), feature_size[0], feature_size[1])
        x_crops_list.append(x_crops)

    return x_crops_list, tlbrs_list
