import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
import argparse
from WSDDN import WSDDN
from torch.utils.tensorboard import SummaryWriter
from dataset import WSDDN_dataset

def drawBoxes(boxes):
    for (x, y, w, h) in boxes:
        plt.hlines(y, x, x + w)
        plt.hlines(y + h, x, x + w)
        plt.vlines(x, y, y + h)
        plt.vlines(x + w, y, y + h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--lr: learning rate\n --epoch: epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--mode', type=str, help='set mode')
    parser.add_argument('--save_dir', type=str, default='saved_models/WSDDN_model', help='Save model with this path')
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    save_dir = args.save_dir
    mode = args.mode

    np.random.seed(3)
    torch.manual_seed(3)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    WSDDN_model = WSDDN()
    WSDDN_model.to(device)

    params = [p for p in WSDDN_model.parameters() if p.requires_grad]
    optimizer1 = torch.optim.SGD(params, lr=lr, momentum=0.9)
    optimizer2 = torch.optim.SGD(params, lr=lr * 0.1, momentum=0.9)

    writer = SummaryWriter('./logs')
    # dummy_data = torch.randint(0, 256, size=(1, 3, 480, 576)).to(torch.float)
    # dummy_rois = np.array([0, 0, 144, 144])
    # for i in range(29):
    #     dummy_rois = np.vstack((dummy_rois, np.array([0, 0, 144, 144])))
    # dummy_rois = torch.tensor(dummy_rois).unsqueeze(0)
    # dummy_label = torch.randint(0, 2, size=(1, 20))
    # writer.add_graph(WSDDN_model, (dummy_data.cuda(), dummy_rois.cuda(), dummy_label.cuda()))

    train_data = WSDDN_dataset('data/VOC2007', 'trainval', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    val_data = WSDDN_dataset('data/VOC2007', 'test', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1)

    if mode == 'train':
        WSDDN_model.train()
        for epoch in range(epochs):
            loss_sum = 0
            iter_num = 0
            print('Epoch: %d' % epoch)
            for i, (img, label, img_info, proposals) in enumerate(train_loader):
                print('Training img:%d\t%s\tNum of proposals:%d\t' % (i, img_info['img_path'], proposals.size(1)))
                if epoch < 10:
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                img = img.to(device)

                result = WSDDN_model(img, proposals, label)
                if not result:
                    print('There are no proposals on this image')
                    continue
                score, loss, rois = result

                loss_sum += loss.item()
                iter_num += 1

                print('loss: %f' % (loss.item()))
                loss.backward()
                if epoch < 10:
                    optimizer1.step()
                else:
                    optimizer2.step()

            writer.add_scalar('Train/loss', loss_sum / iter_num, epoch)
            torch.save(WSDDN_model.state_dict(), save_dir)
        print('Finished training')
        writer.close()
    elif mode == 'test':
        for i, (img, label, img_info, proposals) in enumerate(val_loader):
            proposals = proposals.squeeze()
            scaled_proposals = (proposals / 16).to(torch.int)
            rois = []
            for idx, roi in enumerate(scaled_proposals):
                x = roi[0]
                y = roi[1]
                w = roi[2]
                h = roi[3]
                if w * h < 8:  # Filter small rois
                    continue
                rois.append(proposals[idx])

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            fig.suptitle(img_info['img_path'])
            ax.imshow(img.squeeze().permute(1, 2, 0))
            for x, y, w, h in rois:
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)

            plt.figure('Feature map')
            img = img.to(device)
            feature_map = WSDDN_model.feature_map(img).detach().squeeze()[:64].cpu()
            for i in range(64):
                plt.subplot(8, 8, i + 1)
                plt.imshow(feature_map[i])
            plt.show()
            os.system("pause")