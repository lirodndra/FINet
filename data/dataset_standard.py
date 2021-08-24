import os
import sys


o_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(o_path)
import os.path as osp

import numpy as np
from options.train_options import TrainOptions
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataloaders import Cambridge, SevenScenes


class trans():
    def  __init__(self, fineSize) -> None:
        self.fineSize = fineSize

    def __call__(self, img):
        data_transform = transforms.Compose([
        transforms.Resize(self.fineSize),
        transforms.CenterCrop((self.fineSize, self.fineSize//0.75)),
        transforms.ToTensor()])

        return data_transform(img)


def standard(T):
    opt = TrainOptions().parse()

    data_transform = T(opt.fineSize)
    # dataset loader
    kwargs = dict(train=True, scene=opt.scene, data_path=opt.dataset_dir, transform=data_transform)
    if opt.dataset == '7Scenes':
        dset = SevenScenes(**kwargs)
    elif opt.dataset == 'Cambridge':
        dset = Cambridge(**kwargs)
    else:
        raise NotImplementedError

    # accumulate
    loader = DataLoader(dset, batch_size=opt.batchSize, num_workers=opt.nThreads)
    acc = np.zeros((3, opt.fineSize, int(opt.fineSize // 0.75)))
    c_sq_acc = np.zeros((3, opt.fineSize, int(opt.fineSize // 0.75)))
    for batch_idx, data in enumerate(loader):
        imgs_c = data[0].numpy()
        acc += np.sum(imgs_c, axis=0)
        c_sq_acc += np.sum(imgs_c ** 2, axis=0)

        if batch_idx % 20 == 0:
            print('Accumulated {:d} / {:d}'.format(batch_idx * opt.batchSize, len(dset)))

    N = len(dset) * acc.shape[1] * acc.shape[2]

    mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
    mean_p /= N
    print('Clor images mean pixel = ', mean_p)

    # std = E[x^2] - E[x]^2
    std_p = np.asarray([np.sum(c_sq_acc[c]) for c in range(3)])
    std_p /= N
    std_p -= (mean_p ** 2)
    print('Clor images std. pixel = ', std_p)

    output_filename = osp.join(opt.dataset_dir, opt.dataset, opt.scene, 'stats.txt')
    np.savetxt(output_filename, np.vstack((mean_p, std_p)), fmt='%8.7f')
    print('{:s} written'.format(output_filename))

if __name__ == '__main__':
    standard(trans)
