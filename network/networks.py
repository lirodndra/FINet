import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
    


class FinetGenerator(nn.Module):
    def __init__(self, args):
        super(FinetGenerator, self).__init__()
        self.finetMain = Main(*args)
        self.networks = {'finet_main': self.finetMain}

    def cuda(self, device_id):
        for net in self.networks.values():
            if hasattr(net, 'cuda'):
                net.cuda(device_id)

    def zero_grads(self, ):
        self.optimizers.zero_grad()

    def step_grads(self):
        self.optimizers.step()

    def update_lr(self, new_lr):
        for param_group in self.optimizers.param_groups:
            param_group['lr'] = new_lr

    def save(self, save_path, train_Criterion):
        checkpoint_dict = {}
        for k, net in self.networks.items():
            checkpoint_dict.update({k: net.state_dict()})
        if train_Criterion is not None:
            checkpoint_dict.update({'Criterion': train_Criterion.state_dict()})
        filename = save_path + '.pth'
        torch.save(checkpoint_dict, filename)

    def load(self, save_path, train_Criterion):
        filename = save_path + '.pth'
        nets = torch.load(filename)
        for k, net in self.networks.items():
            net.load_state_dict(nets[k])
        if train_Criterion is not None:
            train_Criterion.load_state_dict(nets['Criterion'])

    def init_optimizers(self, opt, param, lr, betas):
        fea = self.finetMain
        params_list = [{'params': fea.parameters()}]
        if param is not None:
            params_list.append({'params': param})
        self.optimizers = opt(params_list,
                              lr=lr,
                              betas=betas,
                              weight_decay=5e-5)

    def forward(self, input):
        output = self.finetMain(input)
        return output

    def __repr__(self):
        f = self.finetMain
        f_params = sum([p.numel() for p in f.parameters()])
        return repr(f) + '\n\n' + 'Number of total parameters: {:,}'.format(
            f_params) + '\n'


class Main(nn.Module):
    def __init__(self, isTrain, droprate, extracted_layers,
                 hidden_dims, feat_dim):
        super(Main, self).__init__()
        self.droprate = droprate
        self.extracted_layers = extracted_layers
        self.feature_extractor = models.resnet34(pretrained=isTrain)
        self.feature_extractor.avgpool = invariant()
        self.feature_extractor.fc = invariant()

        self.attention = []
        ks = 3
        for ic, _ in self.extracted_layers.items():
            att = TripletAttention(kernel_size=ks)
            self.attention.append(att)
            self.add_module('attention_' + str(ic), att)

        self.dropout = nn.Dropout(self.droprate)
        self.Fusion = Fusion()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_latent = nn.Sequential(nn.Linear(hidden_dims, feat_dim),
                                       nn.Dropout(self.droprate))
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        self.ini_layers = self.attention + [
            self.fc_latent, self.fc_xyz, self.fc_wpqr
        ]

    def forward(self, x):
        outputs = []
        for name, module in self.feature_extractor._modules.items():
            if name not in ['avgpool', 'fc']:
                x = module(x)
            if name in self.extracted_layers.values():
                outputs.append(x)
        self.results = []
        for i, _ in enumerate(self.extracted_layers.keys()):
            x = outputs[i]
            attention = self.attention[i]
            x = attention(x)
            x = self.dropout(x)
            self.results.append(x)

        self.x = self.Fusion(self.results)
        x = [torch.flatten(self.avgpool(x_sub), 1) for x_sub in self.x]
        x = torch.cat(x, 1)
        x = self.fc_latent(x)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)


class Criterion(nn.Module):
    def __init__(self,
                 t_loss_fn=nn.SmoothL1Loss(),
                 q_loss_fn=nn.SmoothL1Loss(),
                 sax=0.0,
                 saq=0.0,
                 learn_beta=False):
        super(Criterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.cuda.FloatTensor([sax]),
                                requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.cuda.FloatTensor([saq]),
                                requires_grad=learn_beta)

    def forward(self, pred, targ):
        ploss = self.t_loss_fn(pred[:, :3], targ[:, :3])
        qloss = self.q_loss_fn(pred[:, 3:], targ[:, 3:])
        loss = torch.exp(-self.sax) * ploss + self.sax + \
               torch.exp(-self.saq) * qloss + self.saq
        return loss, ploss, qloss


##########################################################################
# TripletAttention
# https://github.com/LandskapeAI/triplet-attention
##########################################################################
class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode='circular',
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AxisPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),dim=1)

class AttentionGate(nn.Module):
    def __init__(self, kernel_size):
        super(AttentionGate, self).__init__()
        # kernel_size = 7
        self.compress = AxisPool()
        self.conv = BasicConv(2,
                              1,
                              kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              relu=False)  ###default relu = False

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, kernel_size):  ### default False
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate(kernel_size)
        self.hc = AttentionGate(kernel_size)
        self.hw = AttentionGate(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out = self.hw(x)
        x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        return x_out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        # fusion layers
        self.fusion1 = nn.Conv2d(128,
                                 128,
                                 kernel_size=2,
                                 stride=1, 
                                 bias=False)
        self.fusion2 = nn.Conv2d(256,
                                 256,
                                 kernel_size=2,
                                 stride=1,
                                 groups=2,
                                 bias=False)
        self.fusion3 = nn.Conv2d(512,
                                 512,
                                 kernel_size=2,
                                 stride=1,
                                 groups=4,
                                 bias=False)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

        self.act1 = nn.ELU()
        self.act2 = nn.ELU()
        self.act3 = nn.ELU()

    def _concat_add(self, x, y):
        x = torch.cat([x, x], 1)
        return (x / 2 + y)

    def forward(self, results):
        c1 = results[0]
        c2 = results[1]
        c3 = results[2]

        output_size = c3.shape[-2]
        p1 = F.adaptive_avg_pool2d(c1, output_size)
        p2 = self._concat_add(p1, F.adaptive_avg_pool2d(c2, output_size))
        p3 = self._concat_add(p2, F.adaptive_avg_pool2d(c3, output_size))

        p1 = self.fusion1(p1)
        p2 = self.fusion2(p2)
        p3 = self.fusion3(p3)

        p1 = self.bn1(p1)
        p2 = self.bn2(p2)
        p3 = self.bn3(p3)

        p1 = self.act1(p1)
        p2 = self.act2(p2)
        p3 = self.act3(p3)

        return [p1, p2, p3]


class invariant(nn.Module):
    def __init__(self):
        super(invariant, self).__init__()

    def forward(self, x):
        return x