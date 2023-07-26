import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw


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
    font = ImageFont.truetype('/share/lizhuang05/tmp/simhei.ttf', font_size)
    fillColor = color[::-1]  #  (255,0,0)
    position = pos #  (100,100)
    # if not isinstance(chinese, unicode):
    #     chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL, mode='RGB')
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def plot_tracking(image, boxes, obj_ids, contents=None, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (int(frame_id), fps, len(boxes)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        intbox = tuple(map(int, (x1, y1, x2, y2, x3, y3, x4, y4)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        # if contents is not None:
        #     id_text = id_text + ', {}'.format(contents[i])
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.polylines(im, [np.array(intbox).reshape((-1, 1, 2))], True, color=color, thickness=3)
        # if obj_id == 57:
        #     # 画出 Zouma
        #     cv2.fillPoly(im, [np.array(intbox).reshape((-1, 1, 2))], (255,255,255))
        #     cv2.putText(im, "Zouma", (intbox[0], intbox[5]-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
        im = paint_chinese_opencv(im, id_text, (intbox[0], intbox[1] + 30), color)
    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
