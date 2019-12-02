import selectivesearch
import cv2
import numpy as np
import pickle
import os
import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.io


# def get_proposals(img_path):
#     img = cv2.imread(img_path)
#     img_lbl, proposals = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=20)
#
#     candidates = set()
#     for proposal in proposals:
#         # Excluding same rectangle (with different segments)
#         if proposal['rect'] in candidates:
#             continue
#         x, y, w, h = proposal['rect']
#         if w == 0 or h == 0:
#             continue
#         # distorted rects
#         if w / h > 1.2 or h / w > 1.2:
#             continue
#         candidates.add(proposal['rect'])
#
#     # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     # ax.imshow(img)
#     # for x, y, w, h in candidates:
#     #     rect = mpatches.Rectangle(
#     #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     #     ax.add_patch(rect)
#     #
#     # plt.show()
#
#     return np.array(list(candidates))


def get_proposals(img_path, topN=80):    # Proposals: (x1 y1 x2 y2)
    img = skimage.io.imread(img_path)
    proposals = selective_search.selective_search(img, mode='fast', random=False)
    proposals = selective_search.box_filter(proposals, min_size=20, topN=topN)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(img)
    # for x1, y1, x2, y2 in proposals:
    #     bbox = mpatches.Rectangle(
    #         (x1, y1), (x2 - x1), (y2 - y1), fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(bbox)
    #
    # plt.axis('off')
    # plt.show()
    #
    # os.system('pause')

    return np.array(proposals)


if __name__ == '__main__':
    imgpath2proposals = {}

    dataset_name = 'VOC2007'
    topN = 200
    dataset_path = os.path.join('data', dataset_name, 'JPEGImages')
    imgs_path = os.listdir(dataset_path)

    for i, img_path in enumerate(imgs_path):
        img_path = os.path.join(dataset_path, img_path)
        proposals = get_proposals(img_path, topN=topN)
        imgpath2proposals[img_path] = proposals

        if i % 50 == 0:
            print('%d...' % i)

    f = open(os.path.join('data', '{}_proposals_top{}.pkl'.format(dataset_name, topN)), 'wb')
    pickle.dump(imgpath2proposals, f)
    f.close()