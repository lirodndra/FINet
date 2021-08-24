import os
import os.path as osp
import pickle

import numpy as np
from tools.utils import load_image, process_poses, process_quaternion
from torch.utils import data

# from data.robotcar_sdk.camera_model import CameraModel
# from data.robotcar_sdk.image import load_image as robotcar_loader
# from data.robotcar_sdk.interpolate_poses import (interpolate_ins_poses,
#                                                  interpolate_vo_poses)


class SevenScenes(data.Dataset):
    """
    Initialization Parameter
    ----
    :param scene: scene name ['chess', 'pumpkin', ...]
    :param data_path: root 7scenes data directory.
    Usually '../data/deepslam_data/7Scenes'
    :param train: if True, return the training images. If False, returns the testing images
    :param transform: transform to apply to the images
    :param target_transform: transform to apply to the poses
    :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
    :param real: If True, load poses from SLAM/integration of VO
    :param skip_images: If True, skip loading images and return None instead
    :param vo_lib: Library to use for VO (currently only 'dso')
    """
    def __init__(self,
                 train,
                 scene,
                 data_path,
                 transform=None,
                 target_transform=None,
                 mode=0,
                 seed=7,
                 real=False,
                 skip_images=False,
                 vo_lib='orbslam'):
        self.train = train
        self.mode = 0  # Only read color images
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        data_dir = osp.join(data_path, '7Scenes', scene)

        # decide which sequences to use
        if self.train:
            split_file = osp.join(data_dir, 'train_split.txt')
        else:
            split_file = osp.join(data_dir, 'test_split.txt')
        with open(split_file, 'r') as f:
            seqs = [
                int(l.split('sequence')[-1]) for l in f
                if not l.startswith('#')
            ]

        # read poses and collect image names
        self.c_imgs = []
        self.d_imgs = []
        self.gt_idx = np.empty((0, ), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [
                n for n in os.listdir(osp.join(seq_dir, '.'))
                if n.find('pose') >= 0
            ]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib),
                                     'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)
                frame_idx = pss[:, 0].astype(np.int)
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(
                    seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [
                    np.loadtxt(
                        osp.join(
                            seq_dir,
                            'frame-{:06d}.pose.txt'.format(i))).flatten()[:12]
                    for i in frame_idx
                ]
                ps[seq] = np.asarray(pss)
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack(
                (self.gt_idx,
                 gt_offset + frame_idx))  #list the number of read data
            gt_offset += len(p_filenames)
            c_imgs = [
                osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                for i in frame_idx
            ]
            d_imgs = [
                osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                for i in frame_idx
            ]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if self.train and not real:
            mean_t = np.zeros(
                3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename,
                       np.vstack((mean_t, std_t)),
                       fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq],
                                mean_t=mean_t,
                                std_t=std_t,
                                align_R=vo_stats[seq]['R'],
                                align_t=vo_stats[seq]['t'],
                                align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))

        # q_stats_filename = osp.join(data_dir, 'q_stats.txt')
        # if self.train:
        #     mean_q = np.mean(self.poses[:, 3:], axis=0)
        #     std_q = np.std(self.poses[:, 3:], axis=0) #/ 2
        #     np.savetxt(q_stats_filename, np.vstack((mean_q, std_q)), fmt='%8.7f')
        # else:
        #     mean_q, std_q = np.loadtxt(q_stats_filename)
        # self.poses[:, 3:] -= mean_q
        # self.poses[:, 3:] /= std_q

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 1:
                img = None
                while img is None:
                    img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 2:
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_image(self.c_imgs[index])
                    d_img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            if self.mode == 2:
                img_c = [self.transform(i) for i in img]
            else:
                img_c = self.transform(img)
        bundle = [img_c, pose]

        # if self.train:
        #     img_g = self.transform(img, True)
        #     bundle.append(img_g)

        return bundle

    def __len__(self):
        return self.poses.shape[0]


class Cambridge(data.Dataset):
    """
    Initialization Parameter
    ----
    :param scene: scene name ['GreatCourt', 'KingsCollege', ...]
    :param data_path: root Cambridge data directory.
    Usually '../data/deepslam_data/7Scenes'
    :param train: if True, return the training images. If False, returns the testing images
    :param transform: transform to apply to the images
    :param target_transform: transform to apply to the poses
    :param skip_images: If True, skip loading images and return None instead
    """
    def __init__(self,
                 train,
                 scene,
                 data_path,
                 transform=None,
                 target_transform=None,
                 seed=7,
                 skip_images=False):
        self.train = train
        self.mode = 0  # Only read color images
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        data_dir = osp.join(data_path, 'Cambridge', scene)

        # decide which sequences to use
        if self.train:
            split_file = osp.join(data_dir, 'dataset_train.txt')
        else:
            split_file = osp.join(data_dir, 'dataset_test.txt')
        with open(split_file, 'r') as f:
            pathnpose = [l.split(' ') for l in f][3:]

        # read poses and collect image names
        frame = [p[0] for p in pathnpose]
        self.c_imgs = [osp.join(data_dir, '224', f) for f in frame]
        pss = [p[1:] for p in pathnpose]
        ps = np.asarray(pss, dtype='float')
        # vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if self.train:
            mean_t = np.mean(ps[:, :3], axis=0)
            std_t = np.std(ps[:, :3], axis=0)
            np.savetxt(pose_stats_filename,
                       np.vstack((mean_t, std_t)),
                       fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert pose to translation + log quaternion
        self.poses = np.empty((0, 6))
        pss = process_quaternion(poses_in=ps, mean_t=mean_t, std_t=std_t)
        # align_R=vo_stats['R'],
        # align_t=vo_stats['t'],
        # align_s=vo_stats['s'])
        self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))

        q_stats_filename = osp.join(data_dir, 'q_stats.txt')
        if self.train:
            mean_q = np.mean(self.poses[:, 3:], axis=0)
            std_q = np.std(self.poses[:, 3:], axis=0)
            np.savetxt(q_stats_filename,
                       np.vstack((mean_q, std_q)),
                       fmt='%8.7f')
        else:
            mean_q, std_q = np.loadtxt(q_stats_filename)
        self.poses[:, 3:] -= mean_q
        self.poses[:, 3:] /= std_q

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                img = load_image(self.c_imgs[index])
                pose = self.poses[index]
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img_c = self.transform(img)

        bundle = [img_c, pose]

        return bundle

    def __len__(self):
        return self.poses.shape[0]