import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ROI_Pooling import ROI_Pooling


def bboxes_iou(boxes1, boxes2):    # Calculate the IoUs for pairs of bboxes
    boxes1 = torch.tensor(boxes1)
    boxes2 = torch.tensor(boxes2)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area.to(dtype=float) / (b1_area + b2_area - inter_area).to(dtype=float)

    return iou

class WSDDN(nn.Module):

    def __init__(self, num_classes=20):
        super(WSDDN, self).__init__()
        self.num_classes = num_classes
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:-1])

        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, num_classes)
        self.fc8d = nn.Linear(4096, num_classes)

    def forward(self, img, rois, label):   # img: 1 x C x H x W--tensor, rois: N x 4(x, y, w, h)--Numpy array, label: C(one-hot vector)
        features = self.features(img).squeeze()
        for idx, roi in enumerate(rois):
            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]
            roi_feature = ROI_Pooling(features[:, x: x + w, y: y + h], output_shape=[2, 4]) # Output_shape: 2x4 makes the pooled feature 4096-d
            if idx == 0:
                rois_feature = roi_feature.unsqueeze(0)
            else:
                rois_feature = torch.cat((rois_feature, roi_feature.unsqueeze(0)))

        fc6 = self.fc6(rois_feature)
        fc7 = self.fc7(fc6)

        fc8c = self.fc8c(fc7)
        fc8c = F.softmax(fc8c, dim=1)
        fc8d = self.fc8d(fc7)
        fc8d = F.softmax(fc8d, dim=0)

        scores = fc8c * fc8d
        output = torch.sum(scores, dim=0)

        loss = F.binary_cross_entropy(output, label.to(torch.float32), reduction='none') + self.SpatialRegulariser(scores, label, fc7, rois)
        return loss

    def SpatialRegulariser(self, scores, label, fc7, rois):
        regions_num = rois.shape[0]
        reg_sum = 0
        for k in range(self.num_classes):
            if label[k].item() == 0:
                continue
            sorted_idx = torch.argsort(scores[:, k])

            highest_score_region = rois[sorted_idx[0]]
            highest_score_region_feature = fc7[sorted_idx[0]]

            rest_regions = rois[sorted_idx[1:]]
            rest_regions_features = fc7[sorted_idx[1:]]

            penalise_idx = bboxes_iou(rest_regions, np.tile(highest_score_region, (regions_num - 1, 1)))
            penalise_idx = (penalise_idx > 0.6).to(torch.uint8)

            mask = penalise_idx.unsqueeze(1).repeat(1, 4096)
            rest_regions_features = rest_regions_features * mask

            diff = rest_regions_features - highest_score_region_feature
            diff = diff * scores[:, k][sorted_idx[1:]].view(regions_num - 1, 1)

            reg_sum += torch.pow(diff, 2).sum() * 0.5

        return reg_sum

if __name__ == '__main__':
    test_data = torch.randint(0, 256, size=(1, 3, 224, 224)).to(torch.float)
    test_rois = np.array([0, 0, 2, 9])
    for i in range(29):
        test_rois = np.vstack((test_rois, np.array([0, 0, 2, 9])))
    test_label = torch.randint(0, 2, size=(1, 20)).squeeze()
    WSDDN_model = WSDDN()
    print(WSDDN_model(test_data, test_rois, test_label))