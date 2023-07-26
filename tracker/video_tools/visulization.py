#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visulization.py
@Time    :   2021/04/22 18:09:17
@Author  :   lzneu 
@Version :   1.0
@Contact :   lizhuang05@kuaishou.com
@License :   (C)Copyright 2021-2022, Kwai
@Desc    :   None
'''

# here put the import lib
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image

def paint_chinese_opencv(im, chinese, pos, color, font_size=35):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('/share/lizhuang05/code/video_ocr/video_tools/simhei.ttf', int(font_size))
    fillColor = color[::-1]  #  (255,0,0)
    position = pos #  (100,100)
    # if not isinstance(chinese, unicode):
    #     chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL, mode='RGB')
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def plot_tracking(image, points, content_list, track_id_list, frame_id=0, keyword=""):
    color_table = {"FP": (255, 0, 0), "FN": (0, 0, 255), "TP": (0, 255, 0), "IGNORE": (192, 192, 192)}
    
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(3, max(image.shape[1], image.shape[0]) / 1600.)
    line_thickness = 3 * int(max(image.shape[1], image.shape[0]) / 500.)
    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %s ' % (str(frame_id)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, point in enumerate(points):
        x1, y1,x2, y2, x3, y3, x4, y4 = point
        intbox = list(map(lambda x: int(float(x)), (x1, y1, x2, y2, x3, y3, x4, y4)))
        intbox_np = np.array(intbox).reshape(-1,1,2)
        track_id = int(track_id_list[i])
        content = content_list[i]
        if len(keyword) == 0:
            color = get_color(abs(track_id))
            id_pos = (intbox[0], intbox[1]+30)
        else:
            pos_table = {"FP": (intbox[0], intbox[1]), "FN": (intbox[0], intbox[5]+text_scale*10), "TP": (intbox[0], intbox[1]+text_scale*10), "IGNORE": (intbox[0], intbox[1]+text_scale*10)}
            if keyword in color_table:
                color = color_table[keyword]
                id_pos = pos_table[keyword]
            else:
                color = get_color(int(track_id))
                id_pos = (intbox[0], intbox[1])
        id_text = '{}'.format(int(track_id)) + ":" + content
        # 画四边形
        cv2.polylines(im, [intbox_np], True, color=color, thickness=line_thickness)
        im = paint_chinese_opencv(im, id_text, id_pos, color, font_size=text_scale*10)

    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)
    return image


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
        
    image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb

def plot_detections(image, points, contents=None, scores=None, out_rec = None, color=(0, 255, 0), ids=None):
    im = np.copy(image)
    text_scale = max(3, max(image.shape[1], image.shape[0]) / 1600.)
    line_thickness = 3 * int(max(image.shape[1], image.shape[0]) / 500.)
    
    hotmap = np.zeros(image.shape, dtype=np.float).copy()
    
    for i, bbox in enumerate(points):
        content = ""
        score = -1
        out_rec_one = np.zeros((39,32))
        if contents is not None:
            content = contents[i]
        
        x_min = int(min(bbox[:8][0::2]))
        x_max = int(max(bbox[:8][0::2]))
        y_min = int(min(bbox[:8][1::2]))
        y_max = int(max(bbox[:8][1::2]))
        
        if out_rec is not None:
            try:
                out_rec_one = out_rec[i]
#                 print(out_rec[i])
                out_rec_one = cvt2HeatmapImg(out_rec_one)
                w = int(x_max - x_min)
                h = int(y_max - y_min)

                out_rec_one = cv2.resize(out_rec_one,(w,h))
                hotmap[y_min:y_max,x_min:x_max] = out_rec_one
            except:
                continue
            
#         print(out_rec_one)
        if scores is not None:
            score = scores[i]
        text = '{}# {:.2f}'.format(content, score)
        intbox = list(map(lambda x: int(float(x)), bbox[:8]))
        intbox_np = np.array(intbox).reshape(-1,1,2)
        id_pos = (intbox[0], intbox[1])

#         cv2.polylines(im, [intbox_np], True, color=color, thickness=line_thickness)
#         im = paint_chinese_opencv(im, text, id_pos, color, font_size=text_scale*10)
#     im = im * 0.3 + hotmap * 0.7
    return im,hotmap


def write_detections(points, contents=None, scores=None,res_file=None, color=(0, 255, 0), ids=None):
    
    with open(res_file, 'w') as f:
        if points is None:
            pass
                    
        for i, bbox in enumerate(points):
            content = ""
            score = -1
            if contents is not None:
                content = contents[i]
            if scores is not None:
                score = scores[i]
            text = '{}# {:.2f}'.format(content, score)
            intbox = list(map(lambda x: int(float(x)), bbox[:8]))
            
#             x_min = min(intbox[::2])
#             x_max = max(intbox[::2])
#             y_min = min(intbox[1::2])
#             y_max = max(intbox[1::2])
#             intbox = [x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max]

            strResult = ','.join(
                [str(intbox[0]), str(intbox[1]), str(intbox[2]), str(intbox[3]), str(intbox[4]), str(intbox[5]),
                 str(intbox[6]), str(intbox[7])]) + '\r\n'
                
            f.write(strResult)
