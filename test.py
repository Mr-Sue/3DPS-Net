import os.path
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from models.network import PSNet3D
from models.utils import to_o3d_pcd, yellow, blue
from models.utils import processbar, farthest_point_sample
from functools import partial
from models.dataset import ForestDataset, ForestDataset_collate_fn
from models.kpconv.helper_ply import write_ply

if __name__ == '__main__':

    device = torch.device("cuda:0")

    fcn = PSNet3D()
    fcn.to(device)

    dataset_name = 'forest1'
    output_path = os.path.join('./params', dataset_name, 'output')
    os.makedirs(output_path) if not os.path.exists(output_path) else None

    pretrained = "./params/forest1/epoch_11568_0.99866.pth"
    fcn.load_state_dict(torch.load(pretrained)) if os.path.exists(pretrained) else None

    fcn.eval()

    train_dataset = ForestDataset("/home/user/Desktop/PSNet3D/PSNet3D2",
                                  dataset_name,
                                  split='train',
                                  cut_mode='Height',
                                  grid_size=8.5)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=partial(ForestDataset_collate_fn))

    with torch.no_grad():
        processed = 0
        correct_point_num = 0
        totle_point_num = 0

        key_pts_num = 20

        for batch_idx, (pcd_list, _, labels) in enumerate(loader):
            # for pcd_list, _, _ in loader:
            # pcd_list: [pcd1, pcd2, ...]
            # prompt_ind_list: [ind1, ind2, ...]

            output = os.path.join(output_path, str(loader.dataset.plot_index[batch_idx]) + '.ply')

            processed += 1
            if processed <= 3:
                continue

            pcd_list = pcd_list * key_pts_num
            print(pcd_list[0].shape)
            prompt_ind_list = farthest_point_sample(pcd_list[0][:, :3].unsqueeze(0), key_pts_num)[
                0].cpu().numpy().tolist()

            feats_list = fcn(pcd_list, prompt_ind_list)
            # with open('prediction' + str(loader.dataset.plot_index[batch_idx]) + '.pkl', 'wb') as f:
            #     pickle.dump(feats_list, f)
            print("\rprocess: %s" % (processbar(processed, len(train_dataset))), end="")

            # visualize
            # for i in range(key_pts_num):
            #     pcd = to_o3d_pcd(pcd_list[i][:, :3], blue)
            #     np.asarray(pcd.colors)[feats_list[i].view(-1).round().cpu().numpy() == 1] = np.array(yellow)
            #     np.asarray(pcd.colors)[prompt_ind_list[i]] = np.array([1, 0, 0])
            #     o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="pcd %d" % i)

            fa = np.arange(key_pts_num)


            def find_father(x):
                return x if fa[x] == x else find_father(fa[x])


            iou_thresh = 0.3
            for i in range(key_pts_num):
                j = i + 1
                while j < key_pts_num:

                    mask_i = feats_list[i].view(-1).round()
                    mask_j = feats_list[j].view(-1).round()

                    iou = mask_i + mask_j
                    iou = (iou == 2).sum(dim=0).item() / (iou > 0).sum(dim=0).item()

                    fa_i, fa_j = find_father(i), find_father(j)
                    if iou >= iou_thresh and fa_i != fa_j:
                        fa[fa_i] = fa_j

                    j += 1
            sets = {}
            for i in range(key_pts_num):
                fa_i = find_father(i)
                if sets.get(fa_i) is None:
                    sets[fa_i] = []
                sets[fa_i].append(i)
            keys = sets.keys()
            print("instance num: %d" % len(keys))
            pcd = to_o3d_pcd(pcd_list[0][:, :3], blue)
            label = np.zeros(pcd_list[0].shape[0])
            colors = [np.random.rand(3) for _ in range(len(keys))]
            # 32 x pts_num x 1
            feats_list = torch.stack(feats_list, dim=0)
            item = 0
            for key in keys:
                this_instance_masks_inds = torch.LongTensor(sets[key]).to(device)
                color = colors[item]
                mask = feats_list[this_instance_masks_inds].mean(dim=0).view(-1).round().cpu().numpy()
                np.asarray(pcd.colors)[mask == 1] = color
                item += 1
                label[mask == 1] = item

            write_ply(output, [pcd_list[0][:, :3].numpy(), label], ['x', 'y', 'z', 'instance_label'])
            o3d.visualization.draw_geometries([pcd], width=1000, height=800, window_name="instance seg")
