import os

import torch
import torch.nn as nn

from network.networks import Criterion, FinetGenerator


class Finet(nn.Module):
    def __init__(self, opt):
        super(Finet, self).__init__()

        # self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = opt.save_dir

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.original_A = self.Tensor(opt.batchSize, 3, opt.fineSize,
                                      opt.fineSize)

        print(
            '------------------------ Networks Initializeztion ------------------------'
        )

        # load/define networks
        extracted_layers = {128: 'layer2', 256: 'layer3', 512: 'layer4'}
        args = (opt.isTrain, opt.droprate, extracted_layers,
                sum(extracted_layers.keys()), opt.feat_dim)
        self.netMain = FinetGenerator(args)

        # define loss functions
        self.trainpose_criterion = Criterion(sax=opt.beta,
                                                saq=opt.gamma,
                                                learn_beta=True)
        self.valpose_criterion = Criterion()

        # initialize optimizers
        if self.isTrain:
            self.lr = opt.lr
            self.epochs = opt.epochs
            self.niter_decay = opt.niter_decay
            param = [
                self.trainpose_criterion.sax, self.trainpose_criterion.saq
            ]
            self.netMain.init_optimizers(torch.optim.Adam, param, opt.lr,
                                         (opt.beta1, 0.999))

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netMain, which_epoch,
                              self.trainpose_criterion)

        print(self.netMain)
        print(
            '------------------- Networks Initializeztion Completed -------------------\n\n'
        )

    def set_input(self, input):
        input_A = input[0]
        self.original_A.resize_(input_A.size()).copy_(input_A)
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            loc = input[1].cuda(self.gpu_ids[0])
        else:
            loc = input[1]
        self.target_loc = loc

    def test(self):
        with torch.no_grad():
            output = self.netMain.forward(self.original_A)
        return output

    def eval_losses(self):
        output = self.test()
        loss, ploss, qloss = self.valpose_criterion(output, self.target_loc)
        return output, loss, ploss, qloss

    def optimize_parameters(self):
        pose = self.netMain.forward(self.original_A)
        self.netMain.zero_grads()
        self.loss_total, self.ploss, self.qloss = self.trainpose_criterion(
            pose, self.target_loc)
        self.loss_total.backward()
        self.netMain.step_grads()

    def save_network(self,
                     network,
                     epoch_label,
                     train_Criterion=None):
        save_filename = '%d_finet' % (epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path, train_Criterion)

    def save(self, epoch_label):
        self.save_network(self.netMain, epoch_label,
                          self.trainpose_criterion)

    def load_network(self,
                     network,
                     epoch,
                     train_Criterion=None):
        save_filename = '{}_finet'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load(save_path, train_Criterion)

    def update_hyperparams(self, curr_iter):
        if curr_iter % self.niter_decay == 0:
            decay_frac = 0.95**(curr_iter / self.niter_decay)
            new_lr = self.lr * decay_frac
            self.netMain.update_lr(new_lr)
            print('Updated learning rate: %f' % new_lr)
