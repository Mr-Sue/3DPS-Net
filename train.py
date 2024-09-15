import math
import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.dataset import ForestDataset, ForestDataset_collate_fn
from models.network import PSNet3D
from models.utils import processbar, Logger
from functools import partial


def update_lr(cur_epoch, epoch):
    if cur_epoch < warm_up_epoch:
        lr = base_lr * cur_epoch / warm_up_epoch
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (cur_epoch - warm_up_epoch) / (epoch - warm_up_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    print("\nlr update finished  cur lr: %.8f" % lr)


if __name__ == '__main__':

    device = torch.device("cuda:0")

    fcn = PSNet3D()
    fcn.to(device)

    pretrained = "./params/forest1/epoch_2862_0.99207.pth"
    fcn.load_state_dict(torch.load(pretrained)) if os.path.exists(pretrained) else None

    base_lr = 0.0002
    min_lr = 0.00005
    warm_up_epoch = 0
    optimizer = torch.optim.Adam(fcn.parameters(), lr=base_lr)

    batch_size = 4
    epochs = 500000

    dataset_name = 'forest1'
    save_path = os.path.join('./params', dataset_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None

    train_dataset = ForestDataset(os.path.join("/home/user/Desktop/PSNet3D/PSNet3D2/data", dataset_name),
                                  split='trainval',
                                  cut_mode='Height',
                                  grid_size=8.5)

    loader = DataLoader(train_dataset,
                        batch_size=4,
                        shuffle=True,
                        collate_fn=partial(ForestDataset_collate_fn))

    max_acc = 0
    loss_fn = nn.BCELoss()
    logger = Logger('forest', os.path.join(save_path, 'log.txt'))
    for epoch in range(1, epochs + 1):
        loss_val = 0
        processed = 0
        correct_point_num = 0
        totle_point_num = 0
        for pcd_list, prompt_ind_list, labels in loader:
            labels = labels.float()
            labels = labels.to(device)
            feats_list = fcn(pcd_list, prompt_ind_list)

            pred = torch.cat(feats_list, dim=0).view(-1)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            processed += len(pcd_list)

            pred = pred.round()
            correct_point_num += torch.sum(pred == labels, dim=0).item()
            totle_point_num += labels.shape[0]

            print("\rprocess: %s  loss: %.5f  acc: %.5f" % (
                processbar(processed, len(train_dataset)), loss.item(), correct_point_num / totle_point_num), end="")
        acc = correct_point_num / totle_point_num
        logger.write("\nepoch: %d  train finish, loss: %.5f  acc: %.5f" % (epoch, loss_val, acc))
        val_loss = loss_val
        if max_acc < acc:
            max_acc = acc
            logger.write("save ... ")
            torch.save(fcn.state_dict(), os.path.join(save_path, 'epoch_{}_{:7.5f}.pth'.format(epoch, acc)))
            logger.write("save finish !")

        update_lr(epoch, epochs)
