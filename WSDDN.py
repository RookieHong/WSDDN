import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ROI_Pooling import ROI_Pooling
from torchvision.ops import RoIPool
import traceback


def bboxes_iou(boxes1, boxes2):    # Calculate the IoUs for pairs of bboxes
    # Get the coordinates of bounding boxes (from (x, y, w, h) to (x1, y1, x2, y2))
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 0] + boxes1[:, 2], boxes1[:, 1] + boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 0] + boxes2[:, 2], boxes2[:, 1] + boxes2[:, 3]

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
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        self.feature_map = nn.Sequential(*list(vgg16.features._modules.values())[:-1])

        # self.ROIPool = RoIPool(output_size=(3, 3), spatial_scale=1./16)

        self.fc6 = nn.Linear(4608, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, num_classes)
        self.fc8d = nn.Linear(4096, num_classes)

    def forward(self, img, origin_rois, label):   # img: 1 x C x H x W--tensor float32, rois: 1 x N x 4(x1, y1, x2, y2)--tensor int64, label: 1 x C(one-hot vector)--tensor uint8
        feature_map = self.feature_map(img).squeeze()
        origin_rois = origin_rois.squeeze()

        # # Transform rois to (x1, y1, x2, y2) form
        # origin_rois[:, 2] = origin_rois[:, 0] + origin_rois[:, 2]
        # origin_rois[:, 3] = origin_rois[:, 1] + origin_rois[:, 3]

        # rois = []
        # for i in range(origin_rois.shape[0]):
        #     rois.append(origin_rois[i])
        #
        # rois_feature = self.ROIPool(feature_map, rois)

        feature_map_scale = 16  # VGG16
        scaled_rois = origin_rois / feature_map_scale   # It is still a int tensor after division (Rounding method: floored)

        rois = []
        filtered_origin_rois = []
        for idx, roi in enumerate(scaled_rois):
            x1 = roi[0]
            y1 = roi[1]
            x2 = roi[2]
            y2 = roi[3]
            # if (y2 - y1) * (x2 - x1) < 9:   # Filter small rois
            #     continue
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            roi_feature_map = feature_map[:, y1: y2, x1: x2]
            # if roi_feature_map.size(1) <= 0 or roi_feature_map.size(2) <= 0:
            #     print('roi:(x1: %d y1: %d x2: %d y2: %d) feature map size:(H: %d W: %d)' % (x1, y1, x2, y2, feature_map.size(1), feature_map.size(2)))

            # boxes = torch.tensor([1, x, y, x + w, y + h]).to(torch.device('cuda'))

            # Output_shape: 2x4 or 4x2 makes the pooled feature 4096-d since feature map has 512 channels
            roi_feature = ROI_Pooling(roi_feature_map, output_shape=[3, 3])
            # roi_feature = roi_pool(feature_map, boxes=boxes.to(torch.float), output_size=(7, 7), spatial_scale=feature_map_scale)
            if not rois:
                rois_feature = roi_feature.unsqueeze(0)
            else:
                rois_feature = torch.cat((rois_feature, roi_feature.unsqueeze(0)))

            rois.append((x1, y1, x2, y2))
            filtered_origin_rois.append(origin_rois[idx].cpu().numpy())

        if not rois:
            return False

        filtered_origin_rois = torch.tensor(filtered_origin_rois)

        # rois_feature: N x 4096
        fc6 = self.fc6(rois_feature)    # fc6 outputs: N x 4096
        fc7 = self.fc7(fc6)

        fc8c = self.fc8c(fc7)   # fc8c outputs: N x 20
        fc8c = F.softmax(fc8c, dim=1)
        fc8d = self.fc8d(fc7)   # fc8d outputs: N x 20
        fc8d = F.softmax(fc8d, dim=0)

        scores = fc8c * fc8d    # scores: N x 20
        output = torch.sum(scores, dim=0)

        try:
            loss = F.binary_cross_entropy(output.unsqueeze(0), label.to(torch.float32).cuda(), reduction='mean')
        except RuntimeError:
            print(RuntimeError)
            print('labels:')
            print(label)
            print('outputs:')
            print(output.cpu())
        reg = self.SpatialRegulariser(scores, label, fc7, torch.tensor(rois).cuda())
        loss += reg
        return scores, loss, filtered_origin_rois, reg

    def SpatialRegulariser(self, scores, label, fc7, rois):
        label = label.squeeze()
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

            penalise_idx = bboxes_iou(rest_regions, highest_score_region.repeat(regions_num - 1, 1))
            penalise_idx = (penalise_idx > 0.6).to(torch.int)

            mask = penalise_idx.unsqueeze(1).repeat(1, 4096)
            rest_regions_features = rest_regions_features * mask

            diff = rest_regions_features - highest_score_region_feature
            diff = diff * scores[:, k][sorted_idx[1:]].view(regions_num - 1, 1)

            reg_sum += torch.pow(diff, 2).sum() * 0.5

        return reg_sum / self.num_classes

if __name__ == '__main__':
    dummy_data = torch.randint(0, 256, size=(1, 3, 480, 576)).to(torch.float)
    dummy_rois = np.array([0, 0, 32, 144])
    for i in range(29):
        dummy_rois = np.vstack((dummy_rois, np.array([0, 0, 32, 144])))
    dummy_rois = torch.tensor(dummy_rois).unsqueeze(0)
    dummy_label = torch.randint(0, 2, size=(1, 20))
    WSDDN_model = WSDDN()
    WSDDN_model.to(torch.device('cuda'))
    print(WSDDN_model(dummy_data.cuda(), dummy_rois.cuda(), dummy_label.cuda()))