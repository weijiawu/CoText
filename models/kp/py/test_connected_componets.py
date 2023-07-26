import numpy as np
import cv2
import torch
import kprocess as kp

image = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
    0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0,
    0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0,
    0, 255, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255,
    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255,
    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
    0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
    0, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    0, 0, 255, 255, 255, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
    0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255,
    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
    0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.uint8).reshape([16, 16])

cpu_label_num, cpu_label = cv2.connectedComponents(image, connectivity=4)
image_tensor = torch.from_numpy(image).cuda()
gpu_label = kp.connectedComponents(image_tensor, connectivity=4).to(torch.int32)
gpu_label_num = torch.max(gpu_label).cpu().item() + 1

diff = gpu_label.cpu().numpy() - cpu_label
assert cpu_label_num == gpu_label_num
print(f"diff max: {np.max(np.abs(diff))}")

##################################################################################
region_and_kernel = np.load("region_and_kernel.npy")
region_label_num, region_label = cv2.connectedComponents(region_and_kernel[0], connectivity=4)
kernel_label_num, kernel_label = cv2.connectedComponents(region_and_kernel[1], connectivity=4)

reigon_cuda = torch.from_numpy(region_and_kernel[0]).cuda()
g_region_label = kp.connectedComponents(reigon_cuda, connectivity=4).to(torch.int32)
g_kernel_label = kp.connectedComponents(torch.from_numpy(region_and_kernel[1]).cuda(), connectivity=4).to(torch.int32)

diff_region = region_label - g_region_label.cpu().numpy()
diff_kernel = kernel_label - g_kernel_label.cpu().numpy()

print(f"{np.max(diff_region)}, {np.max(diff_kernel)}")
