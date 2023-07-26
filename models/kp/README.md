# kprocecs: using GPU to accelerate image preprocess with python api

在目标检测等前后处理中，一些算子由于每张图中区域数量不定，不适合做成统一的 batch 处理

torchvision.transform 和 [kornia](https://github.com/kornia/kornia) 提供了一部分基于 torch cuda tensor 计算的 op，性能不是很理想

kprocess 在底层使用 NPP 和  cuda kernel 直接实现，注册为 torch op 方式方便灵活使用


## install 
```python
python setup.py clean && python setup.py bdist_wheel
pip install dist/kprocess-0.1.0-cp36-cp36m-linux_x86_64.whl --force-reinstall
```

## ops
- uint8 灰度图(HW) 和 RGB图 (HWC) warp_perspective , 参考 [test_warp_perspective.py](py/test_warp_perspective.py)
- connected_componets, 图像连通域计算， 参考 [test_connected_componets.py](py/test_connected_componets.py)
- masked_roi， PAN-PP 端到端 ocr 模型，做有 mask 的 roi  crop， 参考 [test_masked_roi.py](py/test_masked_roi.py)
- pixeal_aggreation,  PAN-PP 端到端 ocr 模型, 做 kernel 像素聚类， 参考 [test_pixeal_aggreation.py](py/test_pixeal_aggreation.py)
- minAreaRect，最小外切矩形