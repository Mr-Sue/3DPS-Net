import os
import glob
import numpy as np
import torch
import argparse
import open3d as o3d
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from models.dataset import ForestDataset, ForestDataset_collate_fn
from models.network2 import PSNet3D
from functools import partial
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from models.utils import to_o3d_pcd, yellow, blue, gray, processbar, Logger, farthest_point_sample
from models.kpconv.helper_ply import write_ply, read_ply
import pdb


def select_plot(data, position, grid_size=1):
    x_min = position[0] - grid_size / 2
    y_min = position[1] - grid_size / 2
    x_max = x_min + grid_size
    y_max = y_min + grid_size
    return data[(data[:, 0] >= x_min) & (data[:, 0] <= x_max) & (data[:, 1] >= y_min) & (data[:, 1] <= y_max)]


def data_acquire(path, flag=[0, 1, 2, 3, 4, 5, 6, 7]):
    if path[-4:] == '.txt':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        return np.loadtxt(path, dtype=np.float32)
    elif path[-4:] == '.ply':
        # data: [x, y, z, r, g, b, sem_label, ins_label]
        data = read_ply(path)
        id = data.points.dtype.names
        x = data.points[id[flag[0]]].astype(np.float32)
        y = data.points[id[flag[1]]].astype(np.float32)
        z = data.points[id[flag[2]]].astype(np.float32)
        sem_label = data.points[id[flag[6]]].astype(np.float32)
        ins_label = data.points[id[flag[7]]].astype(np.float32)
        return np.column_stack((x, y, z, np.ones_like(x), np.ones_like(x), np.ones_like(x), sem_label, ins_label))


def pick_points_from_cloud(pcd, required_points=3):
    picked_indices = []
    while len(picked_indices) < required_points:
        # print(f"请在点云窗口中选择点，至少需要选择{required_points}个点。")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        temp_picked = vis.get_picked_points()
        if temp_picked:
            picked_indices.extend(temp_picked)
            # print(f"当前已选择{len(picked_indices)}个点。")
        if len(picked_indices) >= required_points:
            break
        # print("未选择足够的点，请继续选择。")
    return picked_indices[:required_points]

def network_running(points,mask,position_point_data):

    position_point_data[2] = np.average(select_data[:, 2])
    print(position_point_data)
    dis, indices = search_tree2.query([position_point_data], k=1)
    picked_points = np.squeeze(points[indices])
    print(picked_points)
    prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points.to('cpu').numpy())[0][0]
    select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))
    feats = network([select_data], [prompt_ind], device)
    prediction_data = np.array(select_data[:, :3])

    threshold = 0.1
    label = np.concatenate((feats[0] - threshold + 0.5).round().to('cpu').numpy())


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./test2.py --dataset_name=forest1 --pretrained=/home/user/Desktop/PSNet3D/PSNet3D2/params/tree-instance-segmentation.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--dataset_name', default='forest1', type=str, help='The name of folder')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model path')
    parser.add_argument('--position', default=None, type=str, help='position path')
    parser.add_argument('--option', default='prompt', type=str, help='prompt,auto,mix,default:prompt')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    device = torch.device(args.local_rank)

    # ouput_log
    save_path = os.path.join('./params', args.dataset_name)
    os.makedirs(save_path) if not os.path.exists(save_path) else None
    logger = Logger(args.dataset_name, os.path.join(save_path, 'log2.txt'))

    # dataset_loader_configuration
    test_dataset = ForestDataset("/home/user/Desktop/PSNet3D/PSNet3D2",
                                 args.dataset_name,
                                 split='train',
                                 grid_size=8.5)

    test_sampler = DistributedSampler(test_dataset,
                                      num_replicas=dist.get_world_size(),
                                      rank=args.local_rank,
                                      shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=8,
                             collate_fn=partial(ForestDataset_collate_fn))

    network = PSNet3D().cuda(args.local_rank)
    network = DistributedDataParallel(network,
                                      device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)

    if args.pretrained:
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(args.pretrained, map_location='cpu').items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

    network.eval()

    output_path = os.path.join('./params', args.dataset_name, 'output')
    os.makedirs(output_path) if not os.path.exists(output_path) else None

    threshold = 0.1

    with torch.no_grad():

        for batch_idx, (pcd_list, _, labels) in enumerate(test_loader):

            points = pcd_list[0][:, 0:3]
            mask = np.zeros(len(points))

            truth_data = test_loader.dataset.data[batch_idx]
            dataname = test_loader.dataset.datapath[batch_idx]
            filename = dataname.split('/')[-1][:-4]
            
            if args.option != 'auto':
                if args.position:
                    position_file = glob.glob(
                        os.path.join(args.position, 'plot' + dataname.split('/')[-1][-13] + '*ply'))[0]
                    print(position_file)
                    position_points = data_acquire(position_file[0])
                    
                else:
                    print('The position is None')

            search_tree = KDTree(truth_data[:, :3])
            # truth_label = []
            # Mean_Iou = []
            print(len(points))

            # output = os.path.join(output_path, str(test_loader.dataset.plot_index[batch_idx]) + '.ply')
            if args.option == 'prompt':



                print('The position is None')
            elif args.option == 'auto':




                print('The position is None')
            elif args.option == 'mix':
                # position:xyz(not in point cloud)
                for position_point_data in position_points:
                    position_point_data = position_point_data[:3]
                    print(position_point_data)

                # update mask
                if mask.any():
                    points = points[mask]
                    mask = np.zeros(len(points))

                search_tree2 = KDTree(points[:, :3])
                # extract the area
                select_data = select_plot(points, position_point_data, 6)
                # position points z= avearage selected area z
                position_point_data[2] = np.average(select_data[:, 2])
                print(position_point_data)

                # prompt point
                dis, indices = search_tree2.query([position_point_data], k=1)
                picked_points = np.squeeze(points[indices])
                print(picked_points)
                prompt_ind = np.where(select_data.to('cpu').numpy() == picked_points.to('cpu').numpy())[0][0]
                select_data = torch.tensor(np.column_stack((select_data, np.zeros_like(select_data))))

                # network
                feats = network([select_data], [prompt_ind], device)

                select_data_np = np.array(select_data[:, :3])
                label = np.concatenate((feats[0] - threshold + 0.5).round().to('cpu').numpy())
                output_data = prediction_data[label == 1]


                print('The position is None')
            else:
                print('The option is wrong')
                



















