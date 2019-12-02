import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import argparse
from WSDDN import WSDDN
from torchvision.ops import nms
from dataset import WSDDN_dataset

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def drawBoxes(boxes, cls_label, scores):    # boxes: N x 4 (x1, y1, x2, y2)
    color = (random.random(), random.random(), random.random())
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1 - 2, '%s %.2f' % (cls_label, scores[i].item()), bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--params_path', type=str, help='Path to load trained model parameters', default='saved_models/WSDDN_model')
    parser.add_argument('--boxes_per_class', type=int, help='Number of boxes to keep per class per image', default=3)
    parser.add_argument('--threshold', type=float, help='Threshold for the score of boxes', default=0)
    args = parser.parse_args()

    params_path = args.params_path
    boxes_per_class = args.boxes_per_class
    threshold = args.threshold

    np.random.seed(3)
    torch.manual_seed(3)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    WSDDN_model = WSDDN()
    WSDDN_model.load_state_dict(torch.load(params_path))
    WSDDN_model.to(device)
    WSDDN_model.eval()

    val_data = WSDDN_dataset(voc_name='VOC2007', data_type='test', proposals_path='data/VOC2007_proposals_top200.pkl', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for i, (img, label, img_info, proposals) in enumerate(val_loader):
        with torch.no_grad():
            result = WSDDN_model(img.to(device), proposals, label)
        if not result:
            print('There are no proposals on this image')
            continue
        scores, loss, rois, reg = result
        scores = scores.detach()
        rois = rois.detach()

        # # Transform rois to (x1, y1, x2, y2) form
        # rois[:, 2] = rois[:, 0] + rois[:, 2]
        # rois[:, 3] = rois[:, 1] + rois[:, 3]

        plt.figure(img_info['img_path'][0])
        plt.imshow(img.squeeze().permute(1, 2, 0))

        for cls in range(20):
            cls_scores = scores[:, cls]
            # Normalization
            cls_scores = cls_scores / (torch.max(cls_scores) - torch.min(cls_scores))

            filtered_idxes = torch.where(cls_scores > threshold)[0]
            if len(filtered_idxes) <= 0:
                continue

            cls_scores = cls_scores[filtered_idxes]
            selected_rois = rois[filtered_idxes, :]

            # print('Class: %s' % VOC_CLASSES[cls])
            # print(cls_scores)

            nmsed_idxes = py_cpu_nms(selected_rois.cpu().numpy(), cls_scores.cpu().numpy(), thresh=0.4)

            selected_rois = selected_rois[nmsed_idxes[:boxes_per_class], :]
            cls_scores = cls_scores[nmsed_idxes[:boxes_per_class]]

            drawBoxes(selected_rois.cpu().numpy(), VOC_CLASSES[cls], cls_scores)
        plt.show()

        os.system('pause')