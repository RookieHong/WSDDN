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


def drawBoxes(boxes, cls_label):    # boxes: N x 4 (x1, y1, x2, y2)
    color = (random.random(), random.random(), random.random())
    for (x1, y1, x2, y2) in boxes:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1 - 2, cls_label, bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--params_path', type=str, help='Path to load trained model parameters', default='saved_models/WSDDN_model')
    parser.add_argument('--boxes_per_class', type=int, help='Number of boxes to keep per class per image', default=3)
    parser.add_argument('--threshold', type=float, help='Threshold for the score of boxes', default=0.1)
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

    val_data = WSDDN_dataset('data/VOC2007', 'test', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1)

    for i, (img, label, img_info, proposals) in enumerate(val_loader):
        result = WSDDN_model(img.to(device), proposals, label)
        if not result:
            print('There are no proposals on this image')
            continue
        scores, loss, rois = result
        scores = scores.detach()
        rois = rois.detach()

        # Transform rois to (x1, y1, x2, y2) form
        rois[:, 2] = rois[:, 0] + rois[:, 2]
        rois[:, 3] = rois[:, 1] + rois[:, 3]

        plt.figure(img_info['img_path'])
        plt.imshow(img.squeeze().permute(1, 2, 0))

        for cls in range(20):
            filtered_idxes = torch.where(scores[:, cls] > threshold)[0]
            if len(filtered_idxes) <= 0:
                continue
            cls_scores = scores[filtered_idxes, cls]

            print('Class: %s' % VOC_CLASSES[cls])
            print(cls_scores)

            nmsed_idxes = nms(rois, cls_scores, iou_threshold=0.4)
            selected_rois = rois[nmsed_idxes[:boxes_per_class], :]
            drawBoxes(selected_rois.numpy(), VOC_CLASSES[cls])
        plt.show()

        os.system('pause')