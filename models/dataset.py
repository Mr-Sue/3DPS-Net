import pickle

import numpy as np
import os
import json
import torch
from os.path import join as join
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from models.kpconv.helper_ply import read_ply


def crop(data, crop_area, grid_size=1):
    x_min = grid_size * crop_area[0] + np.min(data[:, 0])
    y_min = grid_size * crop_area[1] + np.min(data[:, 1])
    x_max = x_min + grid_size
    y_max = y_min + grid_size
    return data[(data[:, 0] >= x_min) & (data[:, 0] <= x_max) & (data[:, 1] >= y_min) & (data[:, 1] <= y_max)]


def data_acquire(path):
    if path[-4:] == '.txt':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        return np.loadtxt(path, dtype=np.float32)
    elif path[-4:] == '.ply':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        data = read_ply(path)
        x, y, z = data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)
        sem_label = data['semantic_seg'].astype(np.float32)
        ins_label = data['treeID'].astype(np.float32)
        return np.column_stack((x, y, z, np.ones_like(x), np.ones_like(x), np.ones_like(x), sem_label, ins_label))


class ForestDataset(Dataset):
    def __init__(self, root_dir, split='trainval', normal_channel=True, cut_mode='Height', grid_size=3, sem_idx=0):
        """
        初始化数据集
        :param root_dir: 数据集的根目录，应包含train, test, valid三个子目录
        :param split: 选择数据集的分割部分，'train', 'test', 'valid'中的一个
        :param normal_channel: 是否使用额外的通道（如法线或颜色信息）
        :param cut_mode:'Position','Trunk','Height','ALL'
        :param grid_size:cut grid
        :param sem_idx: if the cut_mode is Position' or 'Trunk' , give the semantice label about it
        """
        self.root_dir = root_dir
        self.split = split
        self.normal_channel = normal_channel
        self.datapath = []
        self.grid_size = grid_size

        if split == 'trainval':
            self.datapath = [join(self.root_dir, 'train', filename) for filename in
                             os.listdir(join(self.root_dir, 'train'))] + \
                            [join(self.root_dir, 'val', filename) for filename in
                             os.listdir(join(self.root_dir, 'val'))]
        elif split == 'train' or 'val' or 'test':
            self.datapath = [join(self.root_dir, self.split, filename) for filename in
                             os.listdir(join(self.root_dir, self.split))]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)
        print(self.datapath)

        self.data = []
        for filepath in self.datapath:
            # data: [x, y, z, r, g, b, sem_label, ins_label]
            self.data.append(data_acquire(filepath))

        plot_index_path = os.path.join('/home/user/Desktop/PSNet3D/PSNet3D/params',
                                       root_dir.split('/')[-1] + '_' + str(split) + '_' + str(cut_mode) + '_' + str(
                                           grid_size) + '.pkl')

        if not os.path.exists(plot_index_path):
            self.crop_grid = []
            self.vaild_plot = []
            self.count = 0
            if cut_mode == 'Height':
                for num, one_plot in enumerate(self.data):
                    self.vaild_plot_compute(one_plot, cut_mode, num, 100, grid_size=self.grid_size)

            elif cut_mode == 'Position':
                for num, one_plot in enumerate(self.data):
                    self.vaild_plot_compute(one_plot, cut_mode, num, 1, sem_value=sem_idx, grid_size=self.grid_size)

            elif cut_mode == 'Trunk':
                for num, one_plot in enumerate(self.data):
                    self.vaild_plot_compute(one_plot, cut_mode, num, 20, sem_value=sem_idx, grid_size=self.grid_size)
            elif cut_mode == 'All':
                for num, one_plot in enumerate(self.data):
                    self.vaild_plot_compute(one_plot, cut_mode, num, 0, grid_size=self.grid_size)
            else:
                print('The cut_mode is wrong')

            self.plot_index = {key: value for d in self.vaild_plot for key, value in d.items()}
            with open(plot_index_path, 'wb') as f:
                pickle.dump(self.plot_index, f)
        else:
            with open(plot_index_path, 'rb') as f:
                self.plot_index = pickle.load(f)

        # self.seg_classes = {'Tree': [0, 1, 2, 3]}

    def __getitem__(self, index):
        data_idx = self.plot_index[index][0]
        cloud = self.data[data_idx]
        data_crop_area = self.plot_index[index][1]
        subcloud = crop(cloud, data_crop_area, grid_size=self.grid_size)
        if not self.normal_channel:
            points = subcloud[:, 0:3]  # 仅使用x, y, z
        else:
            points = subcloud[:, 0:6]  # 使用x, y, z, r, g, b
        sem_labels = subcloud[:, -2].astype(np.int32)  # sem标签
        seg_labels = subcloud[:, -1].astype(np.int32)  # 最后一列是seg标签
        # print('======================')
        # print(len(points), self.plot_index[index])

        prompt_point_label = 0

        while prompt_point_label == 0:
            if len(np.unique(seg_labels)) == 1 and seg_labels[0] == 0:
                prompt_point_ind = 0
                break
            # 随机选择一个点作为提示点
            prompt_point_ind = np.random.choice(range(len(points)))
            # 提示点的标签
            prompt_point_label = seg_labels[prompt_point_ind]
            # 生成与提示点具有相同标签的掩码

        label_mask = (seg_labels == prompt_point_label)

        # points(N, 3): (xyz), prompt_point_ind(1): (one label), label_mask(N, )[True/False]
        return points, prompt_point_ind, label_mask


    def __len__(self):
        """
        数据集中的样本数
        """
        return len(self.plot_index)

    def vaild_plot_compute(self, data, mode, num, threshold, sem_value=0, grid_size=1):
        xyz = data[:, :3] - data[:, :3].min(axis=0)
        normalized_points = xyz[:, :2] / (grid_size, grid_size)
        grid_indices = np.floor(normalized_points).astype(np.uint8)

        if mode == 'Height':
            grid_indices = grid_indices[xyz[:, 2] > 3]  # points (z>3) is valid
        elif mode == 'Position':
            grid_indices = grid_indices[data[:, 6] == sem_value]
        elif mode == 'Trunk':
            grid_indices = grid_indices[data[:, 6] == sem_value]

        occupancy_grid = np.zeros((grid_indices[:, 0].max() + 1, grid_indices[:, 1].max() + 1))
        np.add.at(occupancy_grid, (grid_indices[:, 0], grid_indices[:, 1]), 1)
        self.crop_grid.append(occupancy_grid)

        kernel = np.ones((1, 1))
        valid_points = (convolve2d(occupancy_grid, kernel, mode='same') > threshold)

        position_to_id = {(i, j): idx for idx, (i, j) in enumerate(
            (i, j) for i, row in enumerate(valid_points) for j, val in enumerate(row) if val)}
        id_to_position = {idx + self.count: (num, pos) for pos, idx in position_to_id.items()}
        self.count = self.count + len(id_to_position)
        self.vaild_plot.append(id_to_position)


def ForestDataset_collate_fn(batch):
    pcd_list = [torch.from_numpy(points) for points, _, _ in batch]
    prompt_ind_list = [torch.tensor(prompt_point_ind) for _, prompt_point_ind, _ in batch]
    label_list = [torch.from_numpy(label_mask) for _, _, label_mask in batch]
    labels = torch.cat(label_list, dim=0)

    return pcd_list, prompt_ind_list, labels


# ====================================================================================================


class BranchDataset(Dataset):
    def __init__(self, root_dir, split='trainval', normal_channel=True):
        """
        初始化数据集
        :param root_dir: 数据集的根目录，应包含train, test, valid三个子目录
        :param split: 选择数据集的分割部分，'train', 'test', 'valid'中的一个
        :param normal_channel: 是否使用额外的通道（如法线或颜色信息）
        """
        self.root_dir = root_dir
        self.split = split
        self.normal_channel = normal_channel
        self.datapath = []

        if split == 'trainval':
            self.datapath = [join(self.root_dir, 'train', filename) for filename in
                             os.listdir(join(self.root_dir, 'train'))] + \
                            [join(self.root_dir, 'val', filename) for filename in
                             os.listdir(join(self.root_dir, 'val'))]
        elif split == 'train' or 'val' or 'test':
            self.datapath = [join(self.root_dir, self.split, filename) for filename in
                             os.listdir(join(self.root_dir, self.split))]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        self.data = []
        for filepath in self.datapath:
            self.data.append(data_acquire(filepath))

    def __getitem__(self, index):
        cloud = self.data[index]
        if not self.normal_channel:
            points = cloud[:, 0:3]  # 仅使用x, y, z
        else:
            points = cloud[:, 0:6]  # 使用x, y, z, r, g, b
        sem_labels = cloud[:, -2].astype(np.int32)  # sem标签
        seg_labels = cloud[:, -1].astype(np.int32)  # 最后一列是seg标签
        # print('======================')
        # print(len(points), self.plot_index[index])
        # print(np.unique(seg_labels))

        prompt_point_label = 0

        while prompt_point_label == 0:
            if len(np.unique(seg_labels)) == 1 and seg_labels[0] == 0:
                prompt_point_ind = 0
                break
            # 随机选择一个点作为提示点
            prompt_point_ind = np.random.choice(range(len(points)))
            # 提示点的标签
            prompt_point_label = seg_labels[prompt_point_ind]

        # 生成与提示点具有相同标签的掩码
        label_mask = (seg_labels == prompt_point_label)

        return points, prompt_point_ind, label_mask

    def __len__(self):
        """
        数据集中的样本数
        """
        return len(self.data)



def BranchDataset_collate_fn(batch):
    pcd_list = [torch.from_numpy(points) for points, _, _ in batch]
    prompt_ind_list = [torch.tensor(prompt_point_ind) for _, prompt_point_ind, _ in batch]
    label_list = [torch.from_numpy(label_mask) for _, _, label_mask in batch]
    labels = torch.cat(label_list, dim=0)

    return pcd_list, prompt_ind_list, labels


if __name__ == '__main__':
    import open3d as o3d
    from utils import to_o3d_pcd, yellow, blue

    train_dataset = ForestDataset("/home/user/Desktop/PSNet3D/PSNet3D2/data/forest1",
                                  split='trainval',
                                  cut_mode='Height',
                                  grid_size=8.5)

    print(len(train_dataset))
    for i in range(len(train_dataset)):
        point_set, prompt_point_ind, label = train_dataset[i]
        pcd = to_o3d_pcd(point_set[:, :3], blue)
        np.asarray(pcd.colors)[label == 1] = np.array(yellow)
        np.asarray(pcd.colors)[prompt_point_ind] = np.array([1, 0, 0])
        print(point_set.shape)
        o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd")
