from ctypes.wintypes import POINT
import os
import sys
sys.path.insert(1, os.path.abspath('.'))
import numpy as np
import torch
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix
import argparse
import vedo

from mnh.utils_vedo import get_vedo_cameras
from mnh.utils_video import visualize_points_cameras
from mnh.dataset_replica import get_points_from_file

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/TanksAndTemple/llff_room")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--img_w", default="")
    parser.add_argument("--img_h", default="")
    args = parser.parse_args()

    DATA_DIR = args.dataset_dir
    SPLIT = args.split     # train or valid

    train_rot_mat_list = []
    train_trans_mat_list = []

    # load `images.txt` derived from COLMAP
    image_txt = open(os.path.join(DATA_DIR, SPLIT, 'images.txt'), 'r')
    lines = image_txt.readlines()

    for i in range(0, len(lines), 2):      
        s = lines[i].split()

        # convert quaternion to 3x3 rotation matrix
        quaternion = torch.Tensor([
            float(s[1]),
            float(s[2]),
            float(s[3]),
            float(s[4])
        ])
        rot_mat = quaternion_to_matrix(quaternion)
        rot_mat = rot_mat.numpy()

        trans_mat = np.array([
            float(s[5]),
            float(s[6]),
            float(s[7])
        ])

        train_rot_mat_list.append(rot_mat)
        train_trans_mat_list.append(trans_mat)

    train_rot_mat_list = np.array(train_rot_mat_list)
    train_trans_mat_list = np.array(train_trans_mat_list)

    np.save(os.path.join(DATA_DIR, SPLIT, 'R.npy'), train_rot_mat_list)
    np.save(os.path.join(DATA_DIR, SPLIT, 'T.npy'), train_trans_mat_list)

if __name__=='__main__':
    test()