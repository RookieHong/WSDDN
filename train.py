import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import argparse
from WSDDN import WSDDN
from torch.utils.tensorboard import SummaryWriter
from dataset import WSDDN_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--lr: learning rate\n --epoch: epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--GPU', type=int, default=0, help='GPU')
    parser.add_argument('--test', help='test mode')
    parser.add_argument('--save_dir', type=str, default='saved_models/WSDDN_model', help='Save model with this path')
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    save_dir = args.save_dir

    np.random.seed(3)
    torch.manual_seed(3)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    WSDDN_model = WSDDN()
    WSDDN_model.to(device)

    optimizer1 = torch.optim.SGD(WSDDN_model.parameters(), lr=lr, momentum=0.9)
    optimizer2 = torch.optim.SGD(WSDDN_model.parameters(), lr=lr * 0.1, momentum=0.9)

    writer = SummaryWriter('./logs')

    train_data = WSDDN_dataset('data/VOC2007', 'trainval', max_resize_scales=[480, 576, 688, 864, 1200], min_resize=224)
    val_data = WSDDN_dataset('data/VOC2007', 'test', min_resize=224)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=4)
    val_data = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)

    if not args.test:
        WSDDN_model.train()
        for epoch in range(epochs):
            print('Epoch: %d' % epoch)
            for i, (img, label, img_info, proposals) in enumerate(train_loader):
                if epoch < 10:
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                img = img.to(device)

                loss = WSDDN_model(img, proposals, label)
                if not loss:
                    print('Training img %d\tThere are no proposals on this image' % i)
                    continue
                print('Training img %d\t loss: %f' % (i, loss.item()))
                loss.backward()
                if epoch < 10:
                    optimizer1.step()
                else:
                    optimizer2.step()

                writer.add_scalar('Train/loss', loss.item(), epoch)
            torch.save(WSDDN_model.state_dict(), save_dir)
        print('Finished training')
        writer.close()