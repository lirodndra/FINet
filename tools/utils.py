import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import transforms3d.quaternions as txq
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from torchvision.datasets.folder import default_loader


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class DataTransform():
    def __init__(self, train, stats, cropsize=256, color_jitter=0.0):
        t1 = transforms.Resize(cropsize)
        # t2 = transforms.CenterCrop((cropsize, cropsize//0.75)) # keep image size H:W = 3:4
        t3 = transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.5)
        t4 = transforms.ToTensor()
        t5 = transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))
        if train:
            if color_jitter > 0:
                assert color_jitter <= 1.0
                print('Using ColorJitter data augmentation')
                # tc = [t1, t2, t3, t4, t5]
                tc = [t1, t3, t4, t5]
            else:
                print('Not Using ColorJitter')
                # tc = [t1, t2, t4, t5]
                tc = [t1, t4, t5]
            self.data_transform = transforms.Compose(tc)
        else:
            # self.data_transform = transforms.Compose([t1, t3, t4, t5])
            self.data_transform = transforms.Compose([t1, t4, t5])
    
    def __call__(self, data):
        return self.data_transform(data)


def TargeTransform(x):
    return torch.from_numpy(x).float()


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q


def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def process_quaternion(poses_in, mean_t, std_t):#, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, 0:3]

  # align
    for i in range(len(poses_out)):
        q = poses_in[i, 3:]
        q *= np.sign(q[0])   # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        # t = poses_out[i, :3] - align_t
        # poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    # poses_out[:, :3] -= mean_t
    # poses_out[:, :3] /= std_t
    return poses_out


def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


def gkern_2d(size=5, sigma=3):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size//2, size//2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * 3)


def store_pred(pred_poses, targ_poses, idx, batchsize, output, data, pose_s, pose_m):
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = data[1].numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalize the predicted and target translations
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    index = idx + batchsize
    pred_poses[idx:index, :] = output[:batchsize]
    targ_poses[idx:index, :] = target[:batchsize]

    return pred_poses, targ_poses

def qqstore_pred(pred_poses, targ_poses, idx, batchsize, output, data, pose_s, pose_m, q_m, q_s):
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = data[1].numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:] * q_s + q_m) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:] * q_s + q_m) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalize the predicted and target translations
    # output[:, :3] = (output[:, :3] * pose_s) + pose_m
    # target[:, :3] = (target[:, :3] * pose_s) + pose_m

    index = idx + batchsize
    pred_poses[idx:index, :] = output[:batchsize]
    targ_poses[idx:index, :] = target[:batchsize]

    return pred_poses, targ_poses



def cal_losses(pred_poses, targ_poses):
    # loss functions
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

    t_median = np.median(t_loss)
    t_mean = np.mean(t_loss)
    q_median = np.median(q_loss)
    q_mean = np.mean(q_loss)

    print('----------------------------------------------------------------')
    print('Error in translation: \tmedian {:3.2f} m,  \tmean {:3.2f} m\n' \
          'Error in rotation: \tmedian {:3.2f} degrees, \tmean {:3.2f} degree'.format(t_median, t_mean, q_median, q_mean))
    # return t_median, q_median