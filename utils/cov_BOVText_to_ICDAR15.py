"""
https://github.com/xingyizhou/CenterTrack
Modified by weijia wu
"""
import os
import numpy as np
import json
import cv2
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from tqdm import tqdm
def get_annotation(video_path):
    annotation = {}
    with open(video_path,'r',encoding='utf-8-sig') as load_f:
        gt = json.load(load_f)

    for child in gt:
        lines = gt[child]
        annotation.update({child:lines})
    return annotation

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1,2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0]-min_x)**2+(x[1]-min_y)**2)
    start_point = list(_box[0])
    for i in range(0,8,2):
        x,y = box[i],box[i+1]
        if [x,y] == start_point:
            start = i//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return np.array(new_box)


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
#     x1, y1, x2, y2, x3, y3, x4, y4 = vertices
#     lin = []
#     point = []
#     for i in range(4):
#         lin.append(cal_distance(vertices[i*2], vertices[i*2+1], vertices[(i*2+2)%8], vertices[(i*2+3)%8]))
#         point.append([vertices[i*2], vertices[i*2+1], vertices[(i*2+2)%8], vertices[(i*2+3)%8]])
     
#     idx = lin.index(max(lin))
#     a1,b1,a2,b2 = point[idx]
#     if (a2-a1) == 0:
#         angle_interval = 1
#         angle_list = list(range(0, 90, angle_interval))
#     else:
        
#         tan = (b2-b1)/(a2-a1)
#         if tan < 0:
#             angle_interval = 1
#             angle_list = list(range(0, 90, angle_interval))
#         else:
#             angle_interval = 1
#             angle_list = list(range(-90, 0, angle_interval))
            
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    vertices = adjust_box_sort(vertices)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
#     print(v)
#     print(anchor)
    if anchor is None:
#         anchor = v[:, :1]
        anchor = np.array([[v[0].sum()],[v[1].sum()]])/4
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_rotate(box):
    # box : x1,y2...,x3,y3
    theta = find_min_rect_angle(box)
    
    rotate_mat = get_rotate_mat(theta)
    rotated_vertices = rotate_vertices(box, theta)
    x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
    return np.array([x_min, y_min,x_max , y_max]),theta
    
def getBboxesAndLabels_icd13(annotations):
    bboxes = []
    labels = []
    polys = []
    bboxes_ignore = []
    labels_ignore = []
    polys_ignore = []
    IDs = []
    rotates = []
    # points_lists = [] # does not contain the ignored polygons.
    for data in annotations:
#         object_boxes = []
        object_boxes =  [int(float(i)) for i in data["points"]]
        ID = data["ID"]
        content = str(data["transcription"])
        is_caption = str(data["category"])
                    
#         for point in annotation:
#             object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))
        # 获取矩形四个顶点，浮点型
        points = cv2.boxPoints(points).reshape((-1))
        points = [i for i in points]

        if content == "###":
            points.append("###")
        else:
            points.append(content)

        bboxes.append(points)

    return bboxes

def parse_xml(annotation_path,video_path):
    
    bboxess = {}
    annotation = get_annotation(annotation_path)
    for frame_id in annotation.keys():
        frame_name = str(frame_id) + ".jpg"
#         frame_path = os.path.join(params.split(".json")[0],frame_name)
#         print(frame_id)
#         frame_path = os.path.join(video_path,frame_name)
#         try:
#             img = cv2.imread(frame_path)
#             height, width = img.shape[:2]
#         except:
#             print(frame_path+"is None")
#             continue
        bboxes = getBboxesAndLabels_icd13(annotation[frame_id])   
        bboxess.update({frame_id:bboxes})
        
    return bboxess



def get_list(train_data_dir):
    
    img_paths = []
    gt = []
    print("Data preparing...")
    
#     train_list = os.path.join(train_data_dir,"test_list.txt")
#     image_path = os.path.join(train_data_dir,"Frames")
    for cls in os.listdir(train_data_dir):
        video_paths = os.path.join(train_data_dir,cls)
        for video_name in os.listdir(video_paths):
            data = os.path.join(cls,video_name+".json")
            img_paths.append(data)
    return img_paths

if __name__ == '__main__':
    
    # /home/wangjue_Cloud/wuweijia/Code/VideoSpotting/PAN_VTS/pan_pp/eval/Evaluation_minetto# 
    DATA_PATH = '/share/wuweijia/MyBenchMark/MMVText/BOVTextV2/Train'
    OUT_PATH = "/home/wangjue_Cloud/wuweijia/MyBenchMark/MMVText/BOVTextV2/icd15"

    HALF_VIDEO = True
    CREATE_SPLITTED_ANN = True
    CREATE_SPLITTED_DET = True

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    data_path = os.path.join(DATA_PATH, 'Frames')
    ann_path_ = os.path.join(DATA_PATH, 'Annotation')
    seq_list = get_list(os.path.join(DATA_PATH,"Frames"))

    seqs = seq_list
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    for seq in sorted(seqs):
        if '.DS_Store' in seq:
            continue
#             if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
#                 continue
        video_cnt += 1  # video sequence number.
        print(seq.replace("/","_"))
        seq_path = os.path.join(data_path, seq)
        img_path = seq_path.split(".jso")[0]
        ann_path = os.path.join(ann_path_, seq)
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])  # half and half

        image_range = [0, num_images - 1]

        bboxess= parse_xml(ann_path,img_path)
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            out_image = os.path.join(OUT_PATH,"train_image/{}_{}.jpg".format(seq.replace("/","_").split(".")[0],i + 1))
            out_txt = os.path.join(OUT_PATH,"train_gt/{}_{}.txt".format(seq.replace("/","_").split(".")[0],i + 1))
            if os.path.exists(out_image) or os.path.exists(out_txt):
                continue
            if str(i + 1) not in bboxess:
                continue
            img = cv2.imread(os.path.join(img_path, '{}.jpg'.format(i + 1)))
            bboxes = bboxess[str(i + 1)]
            cv2.imwrite(out_image,img)

            with open(out_txt, 'w') as f:
                if bboxes is None:
                    continue

                for bbox in bboxes:
                    strResult = ','.join(
                        [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), str(bbox[4]), str(bbox[5]),
                         str(bbox[6]), str(bbox[7]), str(bbox[8])]) + '\r\n'

                    f.write(strResult)    


