import torch
import kprocess as kp
# torch.ops.load_library("kprocess.cpython-36m-x86_64-linux-gnu.so")

import cv2
import numpy as np

img = cv2.imread("/home/liupeng/remote/kprocess/images/bruce.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("/home/liupeng/remote/kprocess/images/bruce.jpg", cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img).cuda()
print(img_tensor.shape)

# points for test.jpg
src_pts = np.array([
    [125., 150.],
    [562., 40.],
    [562., 282.],
    [54., 328.]
]).astype("float32")

width = 128
height = 64
# coordinate of the points in box points after the rectangle has been
# straightened
dst_pts = np.array([[0, 0],
                    [(width - 1), 0],
                    [(width - 1), height - 1],
                    [0, height - 1]], dtype="float32")


def cpu_warp():
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    return cv2.warpPerspective(img, M, (width, height))


src_quad_tensor = torch.from_numpy(src_pts).float()
dst_quad_tensor = torch.from_numpy(dst_pts).float()
gpu_out = kp.warp_perspective(img_tensor, src_quad_tensor, dst_quad_tensor, height, width, 1).cpu().numpy()
cpu_out = cpu_warp()

diff = np.abs(gpu_out.astype("int32") - cpu_out.astype("int32"))

from PIL import Image

Image.fromarray(gpu_out).save("gpu.pgm")
Image.fromarray(cpu_out).save("cpu.pgm")

"""
python setup.py clean && python setup.py bdist_wheel
pip install dist/kprocess-0.1.0-cp36-cp36m-linux_x86_64.whl --force-reinstall
"""
