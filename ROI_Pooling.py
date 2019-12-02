import torch
import torch.nn as nn
import math


def ROI_Pooling(feature_map, output_shape):   # Feature map: C x h x w, output_shape: h x w
    # channels, height, width = feature_map.size()
    # win_w = math.ceil(width / output_shape[1])
    # win_h = math.ceil(height / output_shape[0])
    # pad_w = math.ceil((output_shape[1] * win_w - width) / 2)
    # pad_h = math.ceil((output_shape[0] * win_h - height) / 2)
    #
    # if pad_h > 0:
    #     # Padding to the top
    #     feature_map = torch.cat((torch.zeros(channels, pad_h, width).cuda(), feature_map), dim=1)
    #     # Padding to the bottom
    #     feature_map = torch.cat((feature_map, torch.zeros(channels, pad_h, width).cuda()), dim=1)
    # if pad_w > 0:
    #     # Padding to the left
    #     feature_map = torch.cat((torch.zeros(channels, height, pad_w).cuda(), feature_map), dim=2)
    #     # Padding to the right
    #     feature_map = torch.cat((feature_map, torch.zeros(channels, height, pad_w).cuda()), dim=2)
    #
    # maxpool = nn.MaxPool2d((win_h, win_w), stride=(win_h, win_w))
    # pooled = maxpool(feature_map)
    ROI_pool = nn.AdaptiveMaxPool2d((output_shape[0], output_shape[1]))
    pooled = ROI_pool(feature_map)
    vector = pooled.view(-1)
    return vector

if __name__ == '__main__':
    tmp = torch.rand((512, 2, 9))
    print(tmp.size())
    print(ROI_Pooling(tmp, output_shape=[2, 4]).size())