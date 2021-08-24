import argparse
import os

import torch
from tools import utils


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # base options
        self.parser.add_argument('--beta', type=float, default=0.0, help='hyper-parameter of pose estimation loss function')
        self.parser.add_argument('--gamma', type=float, default=-3.0, help='hyper-parameter of pose estimation loss function')
        self.parser.add_argument('--dataset_dir', type=str, default='./data')
        self.parser.add_argument('--dataset', type=str, default='7Scenes', help='dataset name')
        self.parser.add_argument('--scene', type=str, default='stairs', help='subsets label of dataset')
        self.parser.add_argument('--seed', type=int, default=7, help='random seed')
        self.parser.add_argument('--skip', type=int, default=10)
        self.parser.add_argument('--variable_skip', type=bool, default=False)
        self.parser.add_argument('--steps', type=int, default=3)
        self.parser.add_argument('--real', type=bool, default=False, help='if True, load poses from SLAM / integration of VO')
        self.parser.add_argument('--logdir', type=str, default='./logs', help='where to save training and evaluation logs')
        self.parser.add_argument('--results_dir', type=str, default='eval', help='where to save evaluation logs')
        self.parser.add_argument('--runs_dir', type=str, default='runs', help='where to save training logs')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--non_local_att', type=bool, default=False, help='if True, choose non-local style self.attention')
        self.parser.add_argument('--fineSize', type=int, default=224, help='then crop to this size')
        self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--feat_dim', type=int, default=2048, help='# dimension of the output feature for pose estimation')
        self.parser.add_argument('--droprate', type=float, default=0.5, help='feature extractor droprate')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=8, type=int, help='threads for loading data and must be less than the number of CPU cores')

        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        self.opt.exp_name = '{:s}_{:s}'.format(self.opt.dataset, self.opt.scene)
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
        self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        self.opt.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset, self.opt.scene)
        utils.mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.results_dir, self.opt.save_dir])

        args = vars(self.opt)
        print('-------------------------------- Options ---------------------------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------------------------ Options End -------------------------------\n')
        
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('-------------------------------- Options ---------------------------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------------------------ Options End -------------------------------\n')
        return self.opt


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--which_epoch', default="fit", help='which epoch to load for inference?')