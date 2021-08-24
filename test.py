import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloaders import Cambridge, SevenScenes
from network.finet import Finet
from options.base_options import TestOptions
from tools.utils import (DataTransform, Logger, TargeTransform, cal_losses,
                         qqstore_pred, store_pred)


def test():
    # Config
    opt = TestOptions().parse()
    opt.batchSize = 1  # test code only supports batchSize = 1
    cuda = torch.cuda.is_available()
    device = "cuda:" + ",".join(str(i) for i in opt.gpu_ids) if cuda else "cpu"

    logfile = osp.join(opt.results_dir, 'eval_log.txt')
    stdout = Logger(logfile)
    print('Logging to {:s}'.format(logfile))
    sys.stdout = stdout

    # Model
    model = Finet(opt)
    model.to(device)
    model.eval()
    print('Loss function hyperparameters: beta={:.4f}, gamma={:.4f}'.format(
        model.trainpose_criterion.sax.item(),
        model.trainpose_criterion.saq.item()))
    stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    # transformer
    # isTrain = True
    data_transform = DataTransform(opt.isTrain, stats, opt.fineSize)
    target_transform = TargeTransform

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene,
                               'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
    if opt.dataset == 'Cambridge':
        pose_stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene,
                                   'q_stats.txt')
        q_m, q_s = np.loadtxt(pose_stats_file)  # mean and stdev

    # Load the dataset
    kwargs = dict(train=opt.isTrain,
                  scene=opt.scene,
                  data_path=opt.dataset_dir,
                  transform=data_transform,
                  target_transform=target_transform,
                  seed=opt.seed)
    if opt.dataset == '7Scenes':
        data_set = SevenScenes(**kwargs)
    elif opt.dataset == 'Cambridge':
        data_set = Cambridge(**kwargs)
    else:
        raise NotImplementedError

    L = len(data_set)
    pred_poses = np.zeros((L, 7))  # store all predicted poses
    targ_poses = np.zeros((L, 7))  # store all target poses

    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

    # inference loop
    n_pred = 0
    for idx, data in enumerate(loader):
        if idx % 200 == 0:
            print('Image {:d} / {:d}'.format(idx, len(loader)))
        model.set_input(data)
        output = model.test()
        # output = output[0]
        batchsize = len(output)
        if opt.dataset == 'Cambridge':
            qqstore_pred(pred_poses, targ_poses, n_pred, batchsize, output,
                         data, pose_s, pose_m, q_m, q_s)
        else:
            store_pred(pred_poses, targ_poses, n_pred, batchsize, output, data,
                       pose_s, pose_m)
        n_pred += batchsize

    # calculate losses
    cal_losses(pred_poses, targ_poses)

    # Save the predicted and true values
    pred_targ = np.hstack((pred_poses, targ_poses))
    csv_expname = opt.dataset + '_' + opt.scene + '_' + 'predtrue'
    csv_filename = osp.join(osp.expanduser(opt.results_dir),
                            '{:s}.csv'.format(csv_expname))
    np.savetxt(csv_filename, pred_targ, delimiter=',')

    pred_poses = (pred_poses[:, :3] - pose_m) / pose_s
    targ_poses = (targ_poses[:, :3] - pose_m) / pose_s

    if opt.dataset == '7Scenes':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        ss = max(1, int(len(data_set) / 500))  # 100 for stairs
        x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
        y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
        z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
        ax.scatter(x[0, :], y[0, :], zs=z[0, :], s=4, c='r')
        ax.scatter(x[1, :], y[1, :], zs=z[1, :], s=4, c='g')
        ax.plot(x[0, :], y[0, :], zs=z[0, :], c='r')
        ax.plot(x[1, :], y[1, :], zs=z[1, :], c='g')
        ax.view_init(azim=119, elev=13)


        plt.show(block=True)
        img_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
        fig.savefig(img_filename)


if __name__ == '__main__':
    test()
