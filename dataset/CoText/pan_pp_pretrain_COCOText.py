import numpy as np
from PIL import Image
from torch.utils import data
import json
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
from os import path as osp
import os 

EPS = 1e-6

ic15_root_dir = '/share/wuweijia/Data/ICDAR2015/'
ic15_train_data_dir = ic15_root_dir + 'train_image/'
ic15_train_gt_dir = ic15_root_dir + 'train_gt/'

coco_root_dir = '/share/wuweijia/Data/COCOTextV2/'
coco_train_data_dir = coco_root_dir + 'train_image/'
coco_train_gt_dir = coco_root_dir + 'train_gt_icd15/'



def get_img(img_path, read_type='cv2'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def check(s):
    for c in s:
        if c in list(string.printable[:-6]):
            continue
        return False
    return True


def get_ann_ic15(img, gt_path, vis=False):
    h, w = img.shape[0:2]
    frames = img.copy()
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = gt[8].replace('\r', '').replace('\n', '')
        if len(word) == 0 or word[0] == '#':
            words.append('###')
        else:
            words.append(word)
        bbox = [int(float(gt[i])) for i in range(8)]
        
        
        if vis:
            cv2.putText(frames, word, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 2)
            points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]],[bbox[6], bbox[7]]], np.int32)
            cv2.polylines(frames, [points], True, (0,0,255), thickness=5)
            
        
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    if vis:
        import random
        frame_id = random.randint(0,1000)
        cv2.imwrite("./outputs/pan_pp_r18_ICDAR15/show/{}.jpg".format(frame_id),frames)
        
    return np.array(bboxes), words


def get_ann_kwai_det(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split('\t')
        word = gt[8].replace('\r', '').replace('\n', '')
        words.append(word)
        bbox = [int(float(gt[i])) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def get_ann_mtwi(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = ",".join(gt[8:]).replace('\r', '').replace('\n', '')
        if len(word) == 0:
            words.append('###')
        else:
            words.append(word)
        bbox = [int(float(gt[i])) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words

 
def random_rot_flip(imgs, mode):
    assert mode in [0,1,2,3]
    if mode == 0: 
        # flip模式
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    else:
        # 旋转模式
        if mode == 1:
            angle = 90
        elif mode == 2:
            angle = 180
        else:
            angle = 270
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
            imgs[i] = img_rotation
    return imgs

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, min_size, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    # print (h_scale, w_scale, h_scale / w_scale)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding_4typing(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs
    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    i = random.randint(0, h - t_h) if h - t_h > 0 else 0
    j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def random_crop_padding(imgs, target_size, crop_word_ratio=0.375):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > crop_word_ratio and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0 # 如果>0说明 文字区域在外面
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0    # 起始x
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0    # 起始y
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask, mask_iou=0.9):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        # 这里设置了只要切割文字超过10% 就mask掉
        if float(np.sum(ind)) / np.sum(ind_before_crop) > mask_iou:
            continue  # 
        word_mask[label] = 0   

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK', use_ctc=False):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    elif voc_type == 'CHINESE':
        with open('/share/lizhuang05/code/pan_pp.pytorch_dev/data/keys.json', encoding='utf-8') as j:
            voc = json.load(j)
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    # voc.append(EOS). # CTC 不存在最后一个位置
    if use_ctc:
        voc.append(UNKNOWN)
        voc.append(PADDING)
    else:
        # attn
        voc.append(EOS)
        voc.append(PADDING)
        voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


class PAN_PP_COCOText(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 with_desc=False,
                 read_type='pil',
                 use_ctc=False,
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform
        self.use_ctc = use_ctc
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.for_rec = with_rec
        self.for_desc = with_desc
        self.read_type = read_type
        self.texts = {}
        self.img_num = 0
        self.img_paths = []
        self.gts = []        

        # ic15
        img_names = [img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(ic15_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = ic15_train_data_dir + img_name
            self.img_paths.append(img_path)

            # gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_name = "gt_" + img_name.split('.')[0] + '.txt'
            gt_path = ic15_train_gt_dir + gt_name
            self.gts.append(gt_path)

        
        # cocotext
        img_names = [img_name for img_name in mmcv.utils.scandir(coco_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(coco_train_data_dir, '.png')])
        for idx, img_name in enumerate(img_names):
            img_path = coco_train_data_dir + img_name
            

            # gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_name = img_name.split('.')[0] + '.txt'
            gt_path = coco_train_gt_dir + gt_name
            if os.path.exists(gt_path):
                self.img_paths.append(img_path)
                self.gts.append(gt_path)


        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE', use_ctc=self.use_ctc)
        self.max_word_num = 200
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return len(self.img_paths)

    def load_ic15_single(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gts[index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_lsvt_single(self, index):
        img_path = self.img_paths['lsvt'][index]
        gt_path = self.gts['lsvt'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_mtwi_single(self, index):
        img_path = self.img_paths['mtwi'][index]
        gt_path = self.gts['mtwi'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_mtwi(img, gt_path)
        return img, bboxes, words, img_path

    def load_kwai_single(self, index):
        img_path = self.img_paths['kwai'][index]
        gt_path = self.gts['kwai'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_kwai_det_single(self, index):
        img_path = self.img_paths['kwai_det'][index]
        gt_path = self.gts['kwai_det'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_width_img_single(self, index):
        img_path = self.img_paths['width_img'][index]
        gt_path = self.gts['width_img'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path

    def load_long_text_single(self, index):
        img_path = self.img_paths['long_text'][index]
        gt_path = self.gts['long_text'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_ic15(img, gt_path)
        return img, bboxes, words, img_path


    # For 自监督
    def __getitem__(self, index): 
        img, bboxes, words, img_path = self.load_ic15_single(index)
        
        data1 = self.__generate_data(img, bboxes, words, img_path)
        data2 = self.__generate_data(img, bboxes, words, img_path)
        return [data1, data2]

    def __generate_data(self, img, bboxes, words, img_path):

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            if word == '???':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if not self.use_ctc:
                if len(word) > self.max_word_len - 1:
                    gt_word[-1] = self.char2id['EOS']
                else:
                    gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:  # 这里放缩到短边为736左右
            img = random_scale(img, self.img_size[0], self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            if type(bboxes) == list:
                for i in range(len(bboxes)):
                    bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                           (bboxes[i].shape[0] // 2, 2)).astype('int32')
            else:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)),
                                    (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.for_rec:
                imgs = random_horizontal_flip(imgs)
            if self.for_desc and random.random() > 0.5:
                gt_instance_before_crop = imgs[1].copy()
                imgs = random_crop_padding_4typing(imgs, self.img_size) 
                img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
                word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask, mask_iou=0.7)

            else:
                imgs = random_rotate(imgs)
                gt_instance_before_crop = imgs[1].copy()
                imgs = random_crop_padding(imgs, self.img_size)     # 在图片中切割, padding边界像素，输出736*736，切长边, 0.825的概率必切文字区域，
                img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
                # 识别+desc 不训练，检测是训练的
                word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask, mask_iou=0.7)  # 切割面积超过0.1的文字, 不训练识别 TODO 改成0.n

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,   # 水平矩形
        )
        if self.for_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask,
                identifies=img_path
            ))
        return data
    
    @staticmethod
    def collate_fn(batch):
        data = []
        for item in batch:
            data += [item[0], item[1]]
        target = {}
        for k in data[0].keys():
            if k == 'identifies':
                img_paths = [item[k] for item in data] # 把batch组到一起
                identifies = torch.zeros((len(img_paths), 1))
                indx_list = list(set(img_paths))
                for i in range(identifies.shape[0]):
                   identifies[i][0] = indx_list.index(img_paths[i])
                target[k] = identifies
            else:
                target[k] = torch.stack([item[k] for item in data]) # 把batch组到一起
        return target


if __name__=='__main__':
    import sys
    sys.path.append('/share/lizhuang05/code/pan_pp.pytorch_dev')
    from utils import visulization

    data_loader = PAN_PP_jointTrain(
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.5,
        read_type='cv2',
        with_rec=True,
        with_desc=True
    )
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        pin_memory=True
    )
    for data in train_loader:
        print(data[0]['identifies'])
        """
        print('-' * 20)
        for k, v in data[0].items():
            if isinstance(v, list):
                print(f'k: {k}, v.shape: {len(v)}')    
            else:
                print(f'k: {k}, v.shape: {v.shape}')
        data1 = data[0]
        data2 = data[1]
        img_id = osp.basename(data1['identifies'][0]).split('.')[0]
        print('绘制增强图像', osp.join('/share/lizhuang05/tmp/',  img_id+'_heamap'))
        visulization.visual_feature(
                    out_path=osp.join('/share/lizhuang05/tmp/', img_id+'_heamap1.jpg'),
                    id2char=data_loader.id2char,
                    instance=data1['gt_instances'],
                    word_masks=data1['word_masks'],
                    bboxes=data1['gt_bboxes'], 
                    identifies=data1['identifies'],
                    gt_words=data1['gt_words'],
                    imgs=data1['imgs'], 
                    gt_texts=data1['gt_texts'], 
                    gt_kernels=data1['gt_kernels'], 
                    training_masks=data1['training_masks']
        )
            

        visulization.visual_feature(
                    out_path=osp.join('/share/lizhuang05/tmp/',img_id+'_heamap2.jpg'),
                    id2char=data_loader.id2char,
                    instance=data2['gt_instances'],
                    word_masks=data2['word_masks'],
                    bboxes=data2['gt_bboxes'], 
                    identifies=data2['identifies'],
                    gt_words=data2['gt_words'],
                    imgs=data2['imgs'], 
                    gt_texts=data2['gt_texts'], 
                    gt_kernels=data2['gt_kernels'], 
                    training_masks=data2['training_masks']
        )
        raise
       """ 
