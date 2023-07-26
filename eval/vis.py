# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import moviepy
import moviepy.video.io.ImageSequenceClip
import shutil
from moviepy.editor import *

def pics2video(frames_dir="", fps=25):
    im_names = os.listdir(frames_dir)
    num_frames = len(im_names)
    frames_path = []
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( frames_dir, str(im_name) + '.jpg')
        frames_path.append(string)
        
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(frames_dir+".mp4", codec='libx264')
#     shutil.rmtree(frames_dir)
    
def Frames2Video(frames_dir=""):
    '''  将frames_dir下面的所有视频帧合成一个视频 '''
    img_root = frames_dir      #'E:\\KSText\\videos_frames\\video_14_6'
    image = cv2.imread(os.path.join(img_root,"1.jpg"))
    h,w,_ = image.shape

    out_root = frames_dir+".avi"
    # Edit each frame's appearing time!
    fps = 20
    fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
    videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
    im_names = os.listdir(img_root)
    num_frames = len(im_names)
    print(len(im_names))
    for im_name in tqdm(range(1, num_frames+1)):
        string = os.path.join( img_root, str(im_name) + '.jpg')
#         print(string)
        frame = cv2.imread(string)
        # frame = cv2.resize(frame, (w, h))
        videoWriter.write(frame)

    videoWriter.release()
    shutil.rmtree(img_root)
    
def get_annotation(video_path):
    annotation = {}
    
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})

    return annotation

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./eval/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

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

if __name__ == "__main__":
    seqs = ["Cls1_Livestreaming","Cls2_Cartoon","Cls3_Sports", "Cls4_Celebrity", "Cls5_Advertising"
          ,"Cls6_NewsReport", "Cls7_Game","Cls8_Comedy","Cls9_Activity","Cls10_Program"
          ,"Cls11_Movie","Cls12_Interview","Cls13_Introduction","Cls14_Talent","Cls15_Photograph"
          ,"Cls16_Government","Cls17_Speech","Cls18_Travel","Cls19_Fashion","Cls20_Campus"
          ,"Cls21_Vlog","Cls22_Driving","Cls23_International","Cls24_Fishery","Cls25_ShortVideo"
          ,"Cls26_Technology","Cls27_Education","Cls28_BeautyIndustry","Cls29_Makeup","Cls30_Dance","Cls31_Eating"]
    
#     dic_name = {
#     'res_video_35':"Video_35_2_3", 
#      'res_video_20':"Video_20_5_1", 
#      'res_video_30':"Video_30_2_3",
#      'res_video_23':"Video_23_5_2", 
#      'res_video_15':"Video_15_4_1", 
#      'res_video_44':"Video_44_6_4", 'res_video_32.json', 'res_video_22.json', 'res_video_24.json', 'res_video_49.json', 'res_video_39.json', 'res_video_11.json', 'res_video_17.json', 'res_video_9.json', 'res_video_55.json', 'res_video_50.json', 'res_video_5.json', 'res_video_48.json', 'res_video_1.json', 'res_video_6.json', 'res_video_53.json', 'res_video_38.json', '.ipynb_checkpoints', 'res_video_34.json', 'res_video_43.json'
#     }
    
    
    
#     root = "/share/wuweijia/MyBenchMark/relabel/To30s/final_MOVText/"
    annotation_path_root ="./outputs/pan_pp_r18_ICDAR15/json"
    gt_annotation_path_root ="./eval/Evaluation_ICDAR13/gt"
#     video_path = "/share/wuweijia/MyBenchMark/relabel/To30s/Video"
    frame_path_root = "/share/wuweijia/Data/ICDAR2015_video/test/frames/"
    result_path_cls_root = "./outputs/pan_pp_r18_ICDAR15/Vis"
    print( os.listdir(annotation_path_root))
    
    dic_name = {}
    sequs = os.listdir(frame_path_root)
    for i in sequs:
        dic_name.update({"res_video_"+i.split("_")[1]:i})
#     dic_name = ["res_video_"+i.split("_")[1] for i in sequs]


    for seq in os.listdir(annotation_path_root):
        if "json" not in seq:
            continue
        if "pynb" in seq:
            continue
        print(seq)
        
        result_path_cls = os.path.join(result_path_cls_root,seq.replace(".json",""))
        annotation_path = os.path.join(annotation_path_root,seq)
        gt_annotation_path = os.path.join(gt_annotation_path_root,seq.replace(".json","_GT.json"))
        
        frame_path = os.path.join(frame_path_root,seq.replace(".json",""))
        
        if not os.path.exists(result_path_cls):
            os.makedirs(result_path_cls)
        print(annotation_path)
        annotation = get_annotation(annotation_path)
        gt_annotation = get_annotation(gt_annotation_path)
        
        gap=1
        lis = np.arange(0,100000,gap)+1
        rgbs={}
        for idx,frame_id in tqdm(enumerate(annotation.keys())):
    #             print(frame_id)

            if int(frame_id) in lis:
                frame_id_ = frame_id
                frame_id_im = frame_id
            else:
                frame_id_ = frame_id
                frame_id_im = frame_id


            while int(frame_id_) not in lis:
                frame_id_ = str(int(frame_id_)-1)
                frame_id_im = frame_id_


    #             frame_name = video_name.split(".json")[0] + "_" + frame_id_im.zfill(6) + ".jpg"
            frame_name = frame_id_im + ".jpg"
            frame_path_1 = os.path.join(frame_path,frame_name)

            frame = cv2.imread(frame_path_1)
            try:
                a = frame.shape[0]
            except:
                print(frame_path_1)
            annotatation_frame = annotation[frame_id_]
            gt_annotation_frame = gt_annotation[frame_id_]
            for ii,data in enumerate(annotatation_frame):
                x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                
                ID = data["ID"]

                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                mask_1 = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask_1, [points], 1)

                if ID in rgbs:
                    frame,rgb = mask_image(frame, mask_1,rgbs[ID])
                else:
                    frame,rgb = mask_image(frame, mask_1)
                    rgbs[ID] = rgb

                r,g,b = rgb[0]
                r,g,b = int(r),int(g),int(b)

                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                cv2.polylines(frame, [points], True, (r,g,b), thickness=5)
                frame=cv2AddChineseText(frame,ID, (int(x1), int(y1) - 20),((0,0,255)), 20)
#                 frame=cv2AddChineseText(frame,id_content, (int(x1), int(y1) - 20),((0,0,255)), 20)
                
            for ii,data in enumerate(gt_annotation_frame):
                x1,y1,x2,y2,x3,y3,x4,y4 =  [int(float(i)) for i in data["points"]]
                ID = data["ID"]
                id_content = str(data["transcription"])
                if "#" in id_content:
                    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    cv2.polylines(frame, [points], True, (0,0,255), thickness=5)
                else:
                    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                    cv2.polylines(frame, [points], True, (0,255,0), thickness=5)
                    
            frame_vis_path = os.path.join(result_path_cls, frame_id+".jpg")
            cv2.imwrite(frame_vis_path, frame)
    #             video_vis_path = "./"
        pics2video(result_path_cls,fps=5)



      
        
        