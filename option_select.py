import pdb
import random

from models.network2 import PSNet3D
import glob

from models.utils import processbar, Logger, farthest_point_sample
from models.utils import to_o3d_pcd, yellow, blue, gray
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from models.kpconv.helper_ply import read_ply, write_ply
import os.path

import numpy as np
import open3d as o3d
import torch
from models.dataset import BranchDataset, BranchDataset_collate_fn
from torch.utils.data import DataLoader
from functools import partial
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree


def IOU(tree, data, truth_data):
    # accurary
    _, indices = tree.query(data, k=1)
    match_data = truth_data[np.concatenate(indices)]
    match_label = np.unique(match_data[:, 7], return_counts=True)

    intersection_count = np.max(match_label[1])

    match_label = int(match_label[0][np.argmax(match_label[1])])
    match_label_tree = truth_data[truth_data[:, 7] == match_label]

    # intersection = match_data[match_data[:, 7] == match_label]
    union = np.unique(np.concatenate((match_label_tree[:, :3], data), axis=0), axis=0)

    Iou = intersection_count / len(union)
    return match_label, Iou


def select_plot(data, position, grid_size=1):
    x_min = position[0] - grid_size / 2
    y_min = position[1] - grid_size / 2
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


def pick_points_from_cloud(pcd, required_points=3):
    picked_indices = []
    while len(picked_indices) < required_points:
        # print(f"请在点云窗口中选择点，至少需要选择{required_points}个点。")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # 这一步会打开一个窗口，让用户可以选择点
        vis.destroy_window()
        temp_picked = vis.get_picked_points()
        if temp_picked:
            picked_indices.extend(temp_picked)
            # print(f"当前已选择{len(picked_indices)}个点。")
        if len(picked_indices) >= required_points:
            break
        # print("未选择足够的点，请继续选择。")
    return picked_indices[:required_points]


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./option_select.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--output', type=str, help='output')
    parser.add_argument('--pretrain', type=str, default=0, help='pretrain')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    device = torch.device(args.local_rank)

    # ouput_log
    logger = Logger('forest', '/home/user/Desktop/PSNet3D/PSNet3D2/params/result.txt')

    # dataset_loader_configuration
    test_dataset = BranchDataset(
        # "/home/user/Desktop/PSNet3D/PSNet3D/data/utilized",
        "/home/user/Desktop/PSNet3D/PSNet3D2/data/data",
        # "/home/user/Desktop/PSNet3D/PSNet3D2/0000/sample0.02",
        split='test')
    test_sampler = DistributedSampler(test_dataset,
                                      num_replicas=dist.get_world_size(),
                                      rank=args.local_rank,
                                      shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=8,
                             collate_fn=partial(BranchDataset_collate_fn))

    # network_configuration
    network = PSNet3D().cuda(args.local_rank)
    # save_path = "./params/tree-instance-segmentation.pth"
    # network.load_state_dict(torch.load(save_path)) if os.path.exists(save_path) else None

    network = DistributedDataParallel(network,
                                      device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)
    # save_path = "/home/user/Desktop/PSNet3D/PSNet3D2/params/tree-instance-segmentation2.pth"
    save_path = args.pretrain
    # save_path = "/home/user/Desktop/PSNet3D/PSNet3D2/params/plot3/epoch_15454_0.99807.pth"

    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(save_path, map_location='cpu').items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    # for name, param in network.named_parameters():
    #     logger.write(str(name))
    #     logger.write(str(param.shape))

    network.eval()
    output_file = args.output
    # output_file = '/home/user/Desktop/PSNet3D/PSNet3D2/params/data/result8'
    logger.write(output_file)
    os.makedirs(output_file) if not os.path.exists(output_file) else None
    with torch.no_grad():
        for batch_idx, (pcd_list, _, labels) in enumerate(test_loader):

            points = pcd_list[0][:, 0:3]
            mask = np.zeros(len(points))

            truth_data = test_loader.dataset.data[batch_idx]

            search_tree = KDTree(truth_data[:, :3])
            truth_label = []
            Mean_Iou = []
            data_file = test_loader.dataset.datapath[batch_idx]
            filename = data_file.split('/')[-1][:-4]
            logger.write('{}'.format(filename))

            # print(len(points))

            for i in range(1000):

                # print(len(points))
                if mask.any():
                    points = points[mask]
                    # print(points)
                    mask = np.zeros(len(points))
                    data = points[points[:, 2] > 3]
                    if data.mean(axis=0)[-1] < 6:
                        break
                # print('-------------------------------------------')
                # print(len(points))
                # pcd = to_o3d_pcd(points, blue)
                # prompt_ind = pick_points_from_cloud(pcd, 1)  # id
                # picked_points = np.asarray(pcd.points)[prompt_ind][0]  # xyz
                picked_points = torch.tensor([0.0, 0.0, 0.0])
                while picked_points[-1] < 6:
                    prompt_ind = np.random.choice(range(len(points)))
                    picked_points = points[prompt_ind]
                    print(picked_points)

                select_data = select_plot(points, picked_points, 8.5)

                prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points.to('cpu').numpy())[0][0]
                select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))

                feats = network([select_data], [prompt_ind], device)

                prediction_data = np.array(select_data[:, :3])

                threshold = 0.1
                label = np.concatenate((feats[0] - threshold + 0.5).round().to('cpu').numpy())

                # visualize
                # pcd = to_o3d_pcd(prediction_data, blue)
                # np.asarray(pcd.colors)[label == 1] = np.array(yellow)
                # np.asarray(pcd.colors)[prompt_ind] = np.array([1, 0, 0])
                # o3d.visualization.draw_geometries([pcd], width=1000, height=800)

                output = os.path.join(output_file, filename + '_' + str(i + 1) + '.ply')
                output_data = prediction_data[label == 1]
                if len(output_data) < 800:
                    # if np.average(output_data, axis=0)[-1] < 6:
                    continue

                # accurary
                match_label, Iou = IOU(search_tree, output_data, truth_data)
                # if match_label == 0:
                #     continue
                # logger.write('The iou of Tree {} corresponds to Tree {} is: {}'.format(i, match_label, Iou))

                # if Iou>0.3:
                truth_label.append(match_label)
                Mean_Iou.append(Iou)

                write_ply(output, [output_data], ['x', 'y', 'z'])
                position_output = np.mean(output_data, axis=0)
                position_output2 = (np.max(output_data, axis=0) + np.min(output_data, axis=0)) * 0.5
                logger.write('{} {} {} {}'.format(str(i + 1),
                                                  position_output[0], position_output[1], position_output[2]))
                # print(len(output_data))
                mask = ~np.isin(points.to('cpu').numpy(), output_data).all(axis=1)

            i = 0
            # logger.write('{}'.format(np.mean(Mean_Iou)))
            # pcd = to_o3d_pcd(points, blue)
            # o3d.visualization.draw_geometries([pcd], width=1000, height=800)
            #
            # confirm = input("comfirm?(y(yes)/n(no)/c(continue)):")
            # while confirm == 'n':
            #     pcd = to_o3d_pcd(points[points[:, 2] > 3], blue)
            #     prompt_ind = pick_points_from_cloud(pcd, 1)  # id
            #     picked_points = np.asarray(pcd.points)[prompt_ind][0]  # xyz
            #     select_data = select_plot(points, picked_points, 8)
            #
            #     prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points)[0][0]
            #     select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))
            #
            #     feats = network([select_data], [prompt_ind], device)
            #
            #     prediction_data = np.array(select_data[:, :3])
            #     threshold = 0.1
            #     label = np.concatenate((feats[0] - threshold + 0.5).round().to('cpu').numpy())
            #     output = os.path.join(output_file, filename + '_' + str(i + 1) + '.ply')
            #     i = i + 1
            #     output_data = prediction_data[label == 1]
            #
            #     if len(output_data) >= 800:
            #         match_label, Iou = IOU(search_tree, output_data, truth_data)
            #         logger.write('The iou of Tree {} corresponds to Tree {} is: {}'.format(i, match_label, Iou))
            #         if match_label != 0:
            #             truth_label.append(match_label)
            #             Mean_Iou.append(Iou)
            #         write_ply(output, [output_data], ['x', 'y', 'z'])
            #         mask = ~np.isin(points.to('cpu').numpy(), prediction_data[label == 1]).all(axis=1)
            #
            #         points = points[mask]
            #
            #     pcdcut = to_o3d_pcd(output_data, blue)
            #     # np.asarray(pcdcut.colors)[label == 1] = np.array(yellow)
            #     # np.asarray(pcdcut.colors)[prompt_ind] = np.array([1, 0, 0])
            #     o3d.visualization.draw_geometries([pcdcut], width=1000, height=800, window_name="pcd %d")
            #
            #     confirm = input("comfirm?(y(yes)/n(no)/c(continue)):")
            # =================================================================================================================

            merge_data = []
            merge_label = []
            merge_rgb = []
            for file in glob.glob(os.path.join(output_file, filename + '*')):
                data = read_ply(file)

                xyz = np.column_stack(
                    (data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)))
                merge_data.append(xyz)

                rgb = np.ones_like(xyz) * (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                merge_rgb.append(rgb)

                ins_label = np.ones(len(data)) * int(file.split('/')[-1].split('_')[-1][:-4])
                merge_label.append(ins_label)

            merge_data = np.vstack(merge_data)
            merge_rgb = np.vstack(merge_rgb).astype(np.uint8)
            _, merge_label = np.unique(np.concatenate(merge_label), return_inverse=True)
            merge_label = merge_label.astype(np.uint8)

            output_merge = os.path.join(output_file, filename + '_merge.ply')

            write_ply(output_merge, [merge_data, merge_rgb, merge_label[:, np.newaxis]], ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])

            # =================================================================================================================

            # merge_data = []
            # merge_label = []
            # merge_rgb = []
            # for file in glob.glob(os.path.join(output_file, filename + '*')):
            #     data = read_ply(file)
            #
            #     xyz = np.column_stack(
            #         (data['x'].astype(np.float32), data['y'].astype(np.float32), data['z'].astype(np.float32)))
            #     merge_data.append(xyz)
            #
            #     rgb = np.ones_like(xyz) * (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            #     merge_rgb.append(rgb)
            #
            #     ins_label = np.ones(len(data)) * int(file.split('/')[-1].split('_')[-1][:-4])
            #     merge_label.append(ins_label)

            # merge_data = np.vstack(merge_data)
            # merge_rgb = np.vstack(merge_rgb).astype(np.uint8)
            # _, merge_label = np.unique(np.concatenate(merge_label), return_inverse=True)
            # merge_label = merge_label.astype(np.uint8)
            # print(np.unique(merge_label))

            # _, indices = search_tree.query(merge_data, k=1)
            # indices = np.squeeze(indices, axis=1)
            # merge_data_output = np.hstack(
            #     (truth_data[:, :3], np.ones((truth_data.shape[0], 3)) * 255, np.zeros((truth_data.shape[0], 1))))
            # merge_data_output[indices] = np.concatenate((merge_data, merge_rgb, merge_label[:, np.newaxis]), axis=1)
            # output_merge = os.path.join(output_file, filename + '_merge.ply')
            # write_ply(output_merge, [merge_data, merge_label], ['x', 'y', 'z', 'instance_label'])
            # write_ply(output_merge, [merge_data_output], ['x', 'y', 'z', 'r', 'g', 'b', 'instance_label'])
            # =================================================================================================================
            logger.write('ok')
            logger.write('Match label:{}'.format(len(truth_label)))
            logger.write('{}'.format(sorted(truth_label)))
            logger.write('Groud Truth:{}'.format(len(np.unique(truth_data[:, 7]))))
            logger.write('{}'.format(np.mean(Mean_Iou)))
