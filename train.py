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
    parser.add_argument('--save_path', type=str, default='saved_models/WSDDN_model', help='Save model with this path')
    parser.add_argument('--load_path', type=str, default='', help='Load model to continue training')
    parser.add_argument('--alpha', help='alpha for spatial regularization', default=0.0001, type=float)
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    save_path = args.save_path
    mode = args.mode
    load_path = args.load_path

    np.random.seed(3)
    torch.manual_seed(3)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    WSDDN_model = WSDDN()
    if not load_path == '':
        WSDDN_model.load_state_dict(torch.load(load_path))
        print('Model %s loaded' % load_path)
    WSDDN_model.to(device)

    params = [p for p in WSDDN_model.parameters() if p.requires_grad]
    optimizer1 = torch.optim.SGD(params, lr=lr, momentum=0.9)
    optimizer2 = torch.optim.SGD(params, lr=0.1 * lr, momentum=0.9)

    writer = SummaryWriter('./logs')
    # dummy_data = torch.randint(0, 256, size=(1, 3, 480, 576)).to(torch.float)
    # dummy_rois = np.array([0, 0, 144, 144])
    # for i in range(29):
    #     dummy_rois = np.vstack((dummy_rois, np.array([0, 0, 144, 144])))
    # dummy_rois = torch.tensor(dummy_rois).unsqueeze(0)
    # dummy_label = torch.randint(0, 2, size=(1, 20))
    # writer.add_graph(WSDDN_model, (dummy_data.cuda(), dummy_rois.cuda(), dummy_label.cuda()))

    train_data = WSDDN_dataset(voc_name='VOC2007', data_type='trainval', proposals_path='data/VOC2007_proposals_top400.pkl', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    val_data = WSDDN_dataset(voc_name='VOC2007', data_type='test', proposals_path='data/VOC2007_proposals_top400.pkl', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1)

    if mode == 'train':
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)

            train_loss_sum = 0
            iter_num = 0
            reg_sum = 0

            # Training
            WSDDN_model.train()
            for i, (img, label, img_info, proposals) in enumerate(train_loader):
                #print('Training img:%d\t%s\tNum of proposals:%d\t' % (i, img_info['img_path'], proposals.size(1)))
                if epoch < (epochs // 2):
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                img = img.to(device)
                proposals = proposals.to(device)

                result = WSDDN_model(img, proposals, label)
                # if not result:
                #     print('There are no proposals on image %d' % i)
                #     continue
                score, loss, rois, reg = result

                train_loss_sum += loss.item()
                iter_num += 1
                reg = reg * args.alpha
                reg_sum += reg.item()

                if i % 100 == 0:
                    print('Trained with %d imgs\tAvg loss: %f\tAvg reg: %f' % (i, train_loss_sum / iter_num, reg_sum / iter_num))
                loss = loss + reg
                loss.backward()
                if epoch < (epochs // 2):
                    optimizer1.step()
                else:
                    optimizer2.step()

            writer.add_scalar('Train/loss', train_loss_sum / iter_num, epoch)

            if epoch % 5 == 0:
                # Validation
                print('Validating...')
                val_loss_sum = 0
                iter_num = 0
                WSDDN_model.eval()
                for i, (img, label, img_info, proposals) in enumerate(val_loader):
                    img = img.to(device)
                    proposals = proposals.to(device)
                    with torch.no_grad():
                        result = WSDDN_model(img, proposals, label)
                        score, loss, rois, reg = result

                    val_loss_sum += loss.item()
                    iter_num += 1
                print('Val loss: %f' % (val_loss_sum / iter_num))

                writer.add_scalar('Val/loss', val_loss_sum / iter_num, epoch)
            torch.save(WSDDN_model.state_dict(), save_path)
        print('Finished training')
        writer.close()
    elif mode == 'test':    # Visualize mid feature maps
        WSDDN_model.eval()

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