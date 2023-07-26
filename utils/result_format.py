import os
import os.path as osp
import zipfile
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw


class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = result_path

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(result_path):
            os.makedirs(result_path)

    def write_result(self, img_name, outputs):
        if 'IC15' in self.data_type:
            self._write_result_ic15(img_name, outputs)
        elif 'TT' in self.data_type:
            self._write_result_tt(img_name, outputs)
        elif 'CTW' in self.data_type:
            self._write_result_ctw(img_name, outputs)
        elif 'MSRA' in self.data_type:
            self._write_result_msra(img_name, outputs)

    def _write_result_ic15(self, img_name, outputs):
        # assert self.result_path.endswith('.zip'), 'Error: ic15 result should be a zip file!'

        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']
        words = None
        if 'words' in outputs:
            words = outputs['words']

        lines = []
        for i, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            if words is None:
                line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
                lines.append(line)
            elif words[i] is not None:
                line = "%d,%d,%d,%d,%d,%d,%d,%d" % tuple(values) + ",%s\n" % words[i]
                lines.append(line)

        file_name = 'res_%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        # z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        # z.write(file_path, file_name)
        # z.close()

    def _write_result_tt(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_ctw(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        tmp_folder = self.result_path.replace('.zip', '')

        file_name = '%s.txt' % image_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()


def _write_result_msra(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ", %d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

def paint_chinese_opencv(im, chinese, pos, color, font_size=35):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('/share/lizhuang05/tmp/simhei.ttf', int(font_size))
    fillColor = color  #  (255,0,0)
    position = pos #  (100,100)
    # if not isinstance(chinese, unicode):
    #     chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL, mode='RGB')
    draw.text(position, chinese, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def plot_bboxs(image, bboxes, contents=None, wscores=None, scores=None, areas=None, color=(0, 255, 0)):
    
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(3, max(image.shape[1], image.shape[0]) / 1000.)
    line_thickness = 1 * int(max(image.shape[1], image.shape[0]) / 400.)
    for i, bbox in enumerate(bboxes):
        content = "###"
        wscore = "" # str(float(wscores[i]))[:4]
        score = "" # str(scores[i])[:4]
        area = "" # str(areas[i])
        if content is not None:
            content = str(contents[i])
        if wscores is not None:
            wscore = str(float(wscores[i]))[:4]
        if scores is not None:
            score = str(scores[i])[:4]
        if areas is not None:
            area = str(areas[i])
        content = "{} | {} | {}| {}".format(content, wscore, score, area)
        x1, y1,x2, y2, x3, y3, x4, y4 = bbox[:8]
        intbox = list(map(lambda x: int(float(x)), (x1, y1, x2, y2, x3, y3, x4, y4)))
        intbox_np = np.array(intbox).reshape(-1,1,2)
        id_pos = (intbox[6], intbox[7])
        # 画四边形
        cv2.polylines(im, [intbox_np], True, color=color, thickness=line_thickness)
        im = paint_chinese_opencv(im, content, id_pos, color, font_size=text_scale*6)
    return im

