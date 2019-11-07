import random
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import selectivesearch
import matplotlib.patches as mpatches

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


def get_proposals(img_path):
    img = cv2.imread(img_path)
    img_lbl, proposals = selectivesearch.selective_search(img, sigma=0.9, min_size=20)

    candidates = set()
    for proposal in proposals:
        # Excluding same rectangle (with different segments)
        if proposal['rect'] in candidates:
            continue
        x, y, w, h = proposal['rect']
        if w == 0 or h == 0:
            continue
        # distorted rects
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(proposal['rect'])

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(img)
    # for x, y, w, h in candidates:
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(rect)
    #
    # plt.show()

    return np.array(list(candidates))


class WSDDN_dataset(data.Dataset):
    def __init__(self, voc_root_dir, data_type, max_resize_scales=None, num_classes=20, min_resize=224):
        assert data_type in ('train', 'val', 'trainval', 'test')
        self.data = []

        self.data_type = data_type
        self.max_resize_scales = max_resize_scales
        self.name2idx = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.num_classes = num_classes
        self.min_resize = min_resize

        self.loadData(voc_root_dir)

    def loadData(self, voc_root_dir):
        for line in open(os.path.join(voc_root_dir, 'ImageSets', 'Main', self.data_type + '.txt')):
            img_data = {}
            img_id = line.strip()
            annotation = ET.parse(os.path.join(voc_root_dir, 'Annotations', img_id + '.xml'))

            label = np.zeros(self.num_classes, dtype=np.uint8)
            gt_bboxes = []
            for obj in annotation.iter('object'):
                cls_name = obj.find('name').text.strip().lower()
                label[self.name2idx[cls_name]] = 1

                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)

                gt_bboxes.append({
                    'bbox': np.array([x1, y1, x2, y2], np.float),
                    'class': cls_name
                })

            img_data['id'] = img_id
            img_data['gt_bboxes'] = gt_bboxes
            img_data['label'] = label
            img_data['img_path'] = os.path.join(voc_root_dir, 'JPEGImages', img_id + '.jpg')

            self.data.append(img_data)

        print('VOC %s Dataset: %s loaded' % (voc_root_dir, self.data_type))

    def __getitem__(self, index):
        img_path = self.data[index]['img_path']
        if self.data[index]['id'] == '007113':
            pass
        trans = []
        img = Image.open(img_path)  # W x H
        proposals = get_proposals(img_path) # x y w h
        # img = cv2.imread(img_path)  # cv2.imread: W x H x C

        # Resize img and proposals to make sure that width or height of img is at least 224 and randomly max length
        if self.max_resize_scales:
            out_img_height = img.size[1]
            out_img_width = img.size[0]

            max_resize = np.random.choice(self.max_resize_scales)
            img_max_len = np.max(img.size)
            max_ratio = max_resize / img_max_len
            if self.min_resize:
                img_min_len = np.min(img.size)
                if img_min_len * max_ratio < self.min_resize:
                    min_ratio = self.min_resize / img_min_len
                    #trans.append(transforms.Resize((int(img.size[1] * min_ratio), int(img.size[0] * min_ratio))))
                    proposals = min_ratio * proposals

                    out_img_height = out_img_height * min_ratio
                    out_img_width = out_img_width * min_ratio

            out_img_height = out_img_height * max_ratio
            out_img_width = out_img_width * max_ratio

            trans.append(transforms.Resize((int(out_img_height), int(out_img_width))))
            proposals = max_ratio * proposals

        trans.append(transforms.RandomHorizontalFlip())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(trans)

        img = transform(img)
        label = self.data[index]['label']
        img_info = self.data[index]
        proposals = proposals.astype(np.int)

        return img, label, img_info, proposals

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    wsddn_dataset = WSDDN_dataset('data/VOC2007', 'train', max_resize_scales=[480, 576, 688, 864, 1200])
    wsddn_loader = DataLoader(dataset=wsddn_dataset, batch_size=1, shuffle=True, num_workers=4)
    print(len(wsddn_loader))
    for img, label, img_info, proposals in wsddn_loader:
        print(label)
        print(img.shape)