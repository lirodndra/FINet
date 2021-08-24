import os
import os.path as osp
import sys

o_path = osp.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(o_path)
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataloaders import Cambridge
from options.base_options import TestOptions
from tools.utils import mkdirs

opt = TestOptions().parse()
base_dir = osp.join(opt.dataset_dir, opt.dataset, opt.scene)
subdir = os.listdir(base_dir)
dirs = [osp.join(base_dir, str(opt.fineSize), p) for p in subdir if 'seq' in p or 'img' in p]
mkdirs(dirs)

def process():
    if not opt.isTrain:
        print('processing VAL data using {:d} cores'.format(opt.nThreads))
    else:
        print('processing TRAIN data using {:d} cores'.format(opt.nThreads))

    # create data loader
    transform = transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(opt.fineSize)])
    dset = Cambridge(scene=opt.scene, data_path=opt.dataset_dir, train=opt.isTrain, transform=transform)
    loader = DataLoader(dset, batch_size=opt.batchSize, num_workers=opt.nThreads)

    # gather information about output filenames
    if not opt.isTrain:
        split_filename = osp.join(base_dir, 'dataset_test.txt')
    else:
        split_filename = osp.join(base_dir, 'dataset_train.txt')
    with open(split_filename, 'r') as f:
        pathnpose = [l.split(' ') for l in f][3:]

    im_filenames = []
    frame = [i[0] for i in pathnpose]
    im_filenames = [osp.join(base_dir, str(opt.fineSize), f) for f in frame]
    assert len(dset) == len(im_filenames)

    # save image
    for batch_idx, (imgs, _) in enumerate(loader):
        for idx, im in enumerate(imgs):
            im_filename = im_filenames[batch_idx * opt.batchSize + idx]
            # im = Image.fromarray(im.numpy())
            try:
                # im.save(im_filename)
                torchvision.utils.save_image(im, im_filename)
            except IOError:
                print('IOError while saving {:s}'.format(im_filename))

        if batch_idx % 10 == 0:
            print('Processed {:d} / {:d}'.format(batch_idx * opt.batchSize, len(dset)))


if __name__ == '__main__':
    ## processing TRAIN data
    opt.isTrain = True
    process()

    ## processing VAL data
    opt.isTrain = False
    process()
