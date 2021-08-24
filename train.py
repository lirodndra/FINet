import os
import sys

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

o_path = os.getcwd()
sys.path.append(o_path)

import datetime
import os.path as osp
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data.dataloaders import Cambridge, SevenScenes
from network.finet import Finet
from options.train_options import TrainOptions
from tools.utils import AverageMeter, DataTransform, Logger, TargeTransform, cal_losses, store_pred, qqstore_pred


def train():
    # Config
    opt = TrainOptions().parse()
    cuda = torch.cuda.is_available()
    device = "cuda:" + ",".join(str(i) for i in opt.gpu_ids) if cuda else "cpu"
    logfile = osp.join(opt.runs_dir, 'log.txt')
    stdout = Logger(logfile)
    print('Logging to {:s}\n'.format(logfile))
    sys.stdout = stdout

    # Model
    model = Finet(opt)
    model.to(device)

    stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)

    train_data_transform = DataTransform(opt.isTrain, stats, opt.fineSize,
                                         opt.color_jitter)
    val_data_transform = DataTransform(False, stats, opt.fineSize,
                                       opt.color_jitter)
    target_transform = TargeTransform

    # Load the dataset
    kwargs = dict(scene=opt.scene,
                  data_path=opt.dataset_dir,
                  target_transform=target_transform,
                  seed=opt.seed)

    if opt.dataset == '7Scenes':
        train_set = SevenScenes(train=True,
                                transform=train_data_transform,
                                **kwargs)
        val_set = SevenScenes(train=False,
                                transform=val_data_transform,
                                **kwargs)
    elif opt.dataset == 'Cambridge':
        train_set = Cambridge(train=True,
                                transform=train_data_transform,
                                **kwargs)
        val_set = Cambridge(train=False,
                            transform=val_data_transform,
                            **kwargs)
    else:
        raise NotImplementedError

    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_set,
                              batch_size=opt.batchSize,
                              shuffle=True,
                              **kwargs)
    val_loader = DataLoader(val_set,
                            batch_size=opt.batchSize,
                            shuffle=False,
                            **kwargs)

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene,
                               'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
    if opt.dataset == 'Cambridge':
        pose_stats_file = osp.join(opt.dataset_dir, opt.dataset, opt.scene,
                                   'q_stats.txt')
        q_m, q_s = np.loadtxt(pose_stats_file)  # mean and stdev

    writer = SummaryWriter(log_dir=opt.runs_dir)
    experiment_name = opt.exp_name
    training_start_time = datetime.datetime.now()

    if opt.which_epoch > 0 and opt.update_lr:
        model.update_hyperparams(opt.which_epoch)

    total_steps = 0
    for epoch in range(opt.which_epoch, opt.epochs + 1):
        epoch_start_time = datetime.datetime.now()
        model.train()
        train_data_time = AverageMeter()
        train_batch_time = AverageMeter()
        end = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_data_time.update(time.time() - end)
            total_steps += 1

            with torch.set_grad_enabled(True):
                model.set_input(data)
                model.optimize_parameters()
            loss_tmp = model.loss_total
            train_batch_time.update(time.time() - end)
            ploss = model.ploss
            qloss = model.qloss

            writer.add_scalar('train/total_err', model.loss_total.item(),
                              total_steps)

            if batch_idx % opt.print_freq == 0:
                print(
                    'Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tTotal loss {:4f}\tploss {:f}\tqloss {:f}'
                    .format(experiment_name, epoch, batch_idx,
                            len(train_loader) - 1, train_data_time.val,
                            train_data_time.avg,
                            train_batch_time.val, train_batch_time.avg,
                            loss_tmp.item(), ploss.item(), qloss.item()))
            end = time.time()

        # Update learning rate
        if opt.update_lr and epoch <= opt.epochs // 2:
            model.update_hyperparams(epoch)

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:
            model.save(epoch)
            print('\nEpoch {:d} checkpoint saved for {:s}\n'.format(
                epoch, experiment_name))

        print('End of epoch %d / %d \t Time Taken: %s\n' %
              (epoch, opt.epochs, datetime.datetime.now() - epoch_start_time))
    writer.close()
    print('\n--------------- Total training time: {} ---------------\n'.format(
        datetime.datetime.now() - training_start_time))


if __name__ == '__main__':
    train()
