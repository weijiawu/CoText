
## 待 cuda 加速 op

### cv2.connectedComponents(image, connectivity=4), 输入 image : uint8 [H, W], 返回 int32 [H, W] 
- cuda 10.2 NPP的实现有 diff

### cv2.minAreaRect(points), [参考](https://github.com/opencv/opencv/blob/master/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.markdown#7b-rotated-rectangle)
- 业务背景：ocr 检测后一张图上有 N 个连通域（变形的 bbox），对这些 bbox 求得最小外切（可旋转）矩形（4 个顶点）
- 最好可实现在一个 kernel launch 中： 输入 一张有 N 个 连通域 label 的图，输出 N 个矩形的 4 个顶点坐标，shape [N, 4， 2] 

```python
如输入是 [16, 16] 的 连通域 label 图， label 范围是 1 - 9, 共输出这个 9 个区域的 minAreaRect 4 个坐标

array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 2, 2, 0, 0, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 4, 4, 4],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 4, 4],
       [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4],
       [0, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 6, 6, 6, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
       [0, 6, 6, 6, 6, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0],
       [0, 0, 6, 6, 6, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0],
       [0, 0, 0, 6, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 8, 8, 8],
       [0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
       [0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
       [0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

---
## 已完成