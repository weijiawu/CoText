import editdistance
import mmcv
import numpy as np
import shapely
from shapely.geometry import Polygon  # 多边形
import Levenshtein
import re

class Corrector:
    def __init__(self,
                 data_type,
                 len_thres,
                 score_thres,
                #  unalpha_score_thres,
                #  ignore_score_thres,
                #  editDist_thres,
                 voc_path=None):
        self.data_type = data_type

        self.len_thres = len_thres
        self.score_thres = score_thres
        # self.unalpha_score_thres = unalpha_score_thres
        # self.ignore_score_thres = ignore_score_thres
        # self.editDist_thres = editDist_thres

        self.voc = self.load_voc(voc_path)

    def process(self, outputs):
        words = outputs['words']
        word_scores = outputs['word_scores']
        words = [self.correct(word, score, self.voc) for word, score in zip(words, word_scores)]
        outputs.update(dict(
            words=words
        ))
        return outputs

    def correct(self, word, score, voc=None):
        EPS = 1e-6
        if len(word) < self.len_thres:
            return None
        if score > self.score_thres:
            return word

        return None

    def load_voc(self, path):
        if path is None:
            return None
        if 'IC15' in self.data_type:
            return self._load_voc_ic15(path)
        elif 'TT' in self.data_type:
            return self._load_voc_tt(path)

    def _load_voc_ic15(self, path):
        lines = mmcv.list_from_file(path)
        voc = []
        for line in lines:
            if len(line) == 0:
                continue
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')
            if line[0] == '#':
                continue
            voc.append(line.lower())
        return voc

    def _load_voc_tt(self, path):
        pass

def eval_img(gt_bboxes, pre_bboxes, gt_words, pre_words, vis=False):

    recall = 0
    precision = 0
    hmean = 0
    acc_list = []

    iouMat = np.empty([1, 1])

    gtPols = []
    detPols = []
    
    gtDontCarePolsNum = []
    detDontCarePolsNum = []
    pairs = []
    detMatched = 0
    detMatchedNums = []
    evaluationLog = ""

    for n in range(len(gt_words)):
        gtPol = gt_bboxes[n]
        gt_words[n] = gt_words[n]
        word = gt_words[n]
        dontCare = word == "###"
        gtPols.append(gtPol)
        if dontCare:
            gtDontCarePolsNum.append(len(gtPols)-1)
    
    evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(
        gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

    for n in range(len(pre_words)):
        detPol = pre_bboxes[n][:8]
        detPols.append(detPol)

        if len(gtDontCarePolsNum) > 0:
            for dontCarePol in gtDontCarePolsNum:
                dontCarePol = gtPols[dontCarePol]
                iop = cal_iop(detPol, dontCarePol)
                if (iop > 0.5):
                    detDontCarePolsNum.append(len(detPols)-1)
                    break
    
    evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(
        detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")
    
    if len(gtPols) > 0 and len(detPols) > 0:
        # Calculate IoU and precision matrixs
        outputShape = [len(gtPols), len(detPols)]
        iouMat = np.empty(outputShape)
        gtRectMat = np.zeros(len(gtPols), np.int8)
        detRectMat = np.zeros(len(detPols), np.int8)
        for gtNum in range(len(gtPols)):
            for detNum in range(len(detPols)):
                pG = gtPols[gtNum]
                pD = detPols[detNum]
                iouMat[gtNum, detNum] = cal_iou(pD, pG)

        for gtNum in range(len(gtPols)):
            for detNum in range(len(detPols)):
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                    if iouMat[gtNum, detNum] > 0.5:
                        gtRectMat[gtNum] = 1
                        detRectMat[detNum] = 1
                        detMatched += 1
                        pairs.append({'gt': gtNum, 'det': detNum})
                        detMatchedNums.append(detNum)
                        gt_word = gt_words[gtNum].upper()
                        pre_word = pre_words[detNum].upper().replace('UNK', '')
                        pre_word = patstr(pre_word, r"a-zA-Z0-9\u4e00-\u9fa5")
                        gt_word = patstr(gt_word, r"a-zA-Z0-9\u4e00-\u9fa5")
                        if Levenshtein.distance(gt_word, pre_word) == 0:
                            acc = 1.0
                        else:
                            acc = 1 - Levenshtein.distance(gt_word, pre_word) / max(len(gt_word), len(pre_word))
                        acc_list.append(acc)
                        evaluationLog += "Match GT #" + \
                            str(gtNum)+gt_word + " with Det #" + str(detNum) +pre_word+ "\n"
    numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
    numDetCare = (len(detPols) - len(detDontCarePolsNum))
    if vis:
        print(evaluationLog)
    # if numGtCare == 0:
    #     recall = float(1)
    #     precision = float(0) if numDetCare > 0 else float(1)
    # else:
    #     recall = float(detMatched) / numGtCare
    #     precision = 0 if numDetCare == 0 else float(
    #         detMatched) / numDetCare
    
    # hmean = 0 if (precision + recall) == 0 else 2.0 * \
    #     precision * recall / (precision + recall)
    # print(precision, recall, hmean, np.mean(np.array(acc_list)))
    return detMatched, numGtCare, numDetCare, acc_list

def cal_max_iou(pre_box, gt_boxes):
    max_iou = 0
    max_index = None
    for i in range(len(gt_boxes)):
        gt = gt_boxes[i][:8]
        iou = cal_iou(pre_box, gt)
        if iou > max_iou:
            max_iou = iou
            max_index = i
    return max_iou, max_index

def cal_iou(pre_line, gt_line):
    a = np.array(pre_line).reshape(-1, 2)  # 多边形二维坐标表示
    poly1 = Polygon(a).convex_hull   # 以左上开始的顺时针坐标对儿
    b = np.array(gt_line).reshape(-1, 2)
    poly2 = Polygon(b).convex_hull
    if not poly1.intersects(poly2):  # 如果不相交
        iou = 0
        return iou
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # 计算两个四边形面积和
            sum_area = poly1.area + poly2.area
            union_area = sum_area - inter_area

            # print(union_area)
            if union_area == 0:
                iou = 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('Warning!:shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def cal_iop(pre_line, gt_line):
    a = np.array(pre_line).reshape(-1, 2)  # 多边形二维坐标表示
    poly1 = Polygon(a).convex_hull   # 以左上开始的顺时针坐标对儿
    b = np.array(gt_line).reshape(-1, 2)
    poly2 = Polygon(b).convex_hull
    if poly1.area <= 0.000001:
        iop = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iop = float(inter_area) / poly1.area
        except shapely.geos.TopologicalError:
            print('Warning!:shapely.geos.TopologicalError occured, iou set to 0')
            iop = 0
    return iop

def patstr(string, pattern_str):
    rule = re.compile(r'[' + pattern_str + ']+')
    tmp_list = rule.findall(string)
    res = ''.join(tmp_list)
    return res