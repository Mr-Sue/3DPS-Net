import pickle

import numpy as np
import os
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
    def __init__(self, root_dir, dataset_name, split='trainval', normal_channel=True, cut_mode='Height', grid_size=3,
                 sem_idx=0):
        """
        初始化数据集
        :param root_dir: root_dir
        :param dataset_name: dataset_name
        :param split: 'train', 'test', 'val','trainval'
        :param normal_channel: rgb
        :param cut_mode:'Position','Height','ALL'
        :param grid_size:cut grid
        :param sem_idx: if the cut_mode is Position' or 'Trunk' , give the semantice label about it
        """
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, 'data', dataset_name)
        self.split = split
        self.normal_channel = normal_channel
        self.datapath = []
        self.grid_size = grid_size

        if split == 'trainval':
            self.datapath = [join(self.dataset_dir, 'train', filename) for filename in
                             os.listdir(join(self.dataset_dir, 'train'))] + \
                            [join(self.dataset_dir, 'val', filename) for filename in
                             os.listdir(join(self.dataset_dir, 'val'))]
        elif split == 'train' or 'val' or 'test':
            self.datapath = [join(self.dataset_dir, self.split, filename) for filename in
                             os.listdir(join(self.dataset_dir, self.split))]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        self.data = []
        for filepath in self.datapath:
            # data: [x, y, z, r, g, b, sem_label, ins_label]
            self.data.append(data_acquire(filepath))

        plot_index_path = os.path.join(root_dir, 'params', 'dataset',
                                       dataset_name + '_' + str(split) + '_' + str(grid_size) + '.pkl')

        if not os.path.exists(plot_index_path):
            self.crop_grid = []
            self.vaild_plot = []
            self.count = 0

            for num, one_plot in enumerate(self.data):
                self.vaild_plot_compute(one_plot, num, 100, grid_size=self.grid_size)

            self.plot_index = {key: value for d in self.vaild_plot for key, value in d.items()}
            with open(plot_index_path, 'wb') as f:
                pickle.dump(self.plot_index, f)
        else:
            with open(plot_index_path, 'rb') as f:
                self.plot_index = pickle.load(f)

    def __getitem__(self, index):
        data_idx = self.plot_index[index][0]
        cloud = self.data[data_idx]
        data_crop_area = self.plot_index[index][1]
        subcloud = crop(cloud, data_crop_area, grid_size=self.grid_size)
        if not self.normal_channel:
            points = subcloud[:, 0:3]  # x, y, z
        else:
            points = subcloud[:, 0:6]  # x, y, z, r, g, b
        sem_labels = subcloud[:, -2].astype(np.int32)  # semantic label
        seg_labels = subcloud[:, -1].astype(np.int32)  # segmentation label

        prompt_point_label = 0

        while prompt_point_label == 0:
            if len(np.unique(seg_labels)) == 1 and seg_labels[0] == 0:
                prompt_point_ind = 0
                break
            # random select one point
            prompt_point_ind = np.random.choice(range(len(points)))
            # acquire the segmentation label of selected point
            prompt_point_label = seg_labels[prompt_point_ind]

        # generate the mask with same label
        label_mask = (seg_labels == prompt_point_label)

        # points(N, 3): (xyz), prompt_point_ind(1): (one label), label_mask(N, )(True / False)
        return points, prompt_point_ind, label_mask

    def __len__(self):
        """
        数据集中的样本数
        """
        return len(self.plot_index)

    def vaild_plot_compute(self, data, num, threshold, grid_size=1):
        xyz = data[:, :3] - data[:, :3].min(axis=0)
        normalized_points = xyz[:, :2] / (grid_size, grid_size)
        grid_indices = np.floor(normalized_points).astype(np.uint8)

        grid_indices = grid_indices[xyz[:, 2] > 3]  # points (z>3) is valid

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


if __name__ == '__main__':
    import open3d as o3d
    from utils import to_o3d_pcd, yellow, blue

    train_dataset = ForestDataset("/home/user/Desktop/PSNet3D/PSNet3D2",
                                  'forest1',
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
