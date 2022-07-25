import builtins
import os
import sys
import time
import argparse
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter


from PIL import ImageFilter
from util import adjust_learning_rate, AverageMeter, subset_classes, get_activation
import models.resnet as resnet
from tools import get_logger
import graph_blocks.builder

import cv2
from PIL import Image

cv2.setNumThreads(0)
memorycache = False
try:
    import mc, io
    memorycache = True
    print("using memory cache")
except:
    print("missing memory cache")
    pass

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'],
                        help='use full or subset of the dataset')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')
    parser.add_argument('--restart', action='store_true', help='whether to restart')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

    # model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--mem_bank_size', type=int, default=128000)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')

    parser.add_argument('--weights', type=str, help='weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--checkpoint_path', default='output/mean_shift_default', type=str,
                        help='where to save checkpoints. ')

    # GCN_configs
    parser.add_argument('--GCN_configs', type=str)


    opt = parser.parse_args()

    with open(opt.GCN_configs) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt.GCN_configs_q = config['GCN_configs_q']
    opt.GCN_configs_t = config['GCN_configs_t']

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageFolderEx, self).__init__(root)
        self.transform = transform
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        sample = self.transform(sample)

        return index, sample, target

    def __len__(self):
        return len(self.samples)

# Extended version of ImageFolder to return index of image too.
class Image100FolderEx(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.class_list = '/mnt/lustre/share/tangshixiang/data/ImageNet/ImageNet100_class_map.txt'
        super(Image100FolderEx, self).__init__(root)
        self.transform = transform
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _find_classes(self, dir):
        with open(self.class_list, 'r') as fp:
            classes = fp.read().strip().split("\n")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        sample = self.transform(sample)

        return index, sample, target

    def __len__(self):
        return len(self.samples)


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp

class GraphNeruralNetworkLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, withweight=True,
                withbn=True, activation_type='relu'):
        super(GraphNeruralNetworkLayer, self).__init__()
        if withweight:
            self.fc = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.register_buffer('fc', None)

        self.sigma = get_activation(activation_type)

        if withbn:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.register_buffer('bn', None)

    def forward(self, input, adj=None):
        if self.fc is not None:
            support = self.fc(input)
        else:
            support = input

        if adj is not None:
            output = torch.mm(adj, support)
        else:
            output = support

        if self.bn is not None:
            output = self.bn(output)

        return self.sigma(output)

class MultiLayerGraphNeuralNetwork(nn.Module):
    def __init__(self, GCN_configs):
        super(MultiLayerGraphNeuralNetwork, self).__init__()
        activation_type_list = {'relu': F.relu, 'identity': nn.Identity(), 'softmax': F.softmax}
        self.GCN_lists = nn.ModuleList()
        num_layers = len(GCN_configs)
        for i in range(num_layers):
            # GCN_configs[i]['activation'] = activation_type_list[GCN_configs[i]['activation_type']]
            # del GCN_configs[i]['activation_type']
            self.GCN_lists.append(GraphNeruralNetworkLayer(**GCN_configs[i]))

    def forward(self, input, adj=None):
        for gc in self.GCN_lists:
            input = gc(input, adj=adj)
        return input

class MeanShift(nn.Module):
    def __init__(self, arch, m=0.99, mem_bank_size=128000, topk=5, GCN_configs_q=None, GCN_configs_t=None):
        super(MeanShift, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk

        # create encoders and projection layers
        # both encoders should have same arch
        if 'resnet' in arch:
            self.encoder_q = resnet.__dict__[arch]()
            self.encoder_t = resnet.__dict__[arch]()

        # save output embedding dimensions
        # assuming that both encoders have same dim
        feat_dim = self.encoder_q.fc.in_features # feat_dim = 2048
        hidden_dim = feat_dim * 2 # hidden_dim = 4096
        proj_dim = feat_dim // 4 # proj_dim = 512

        # projection layers
        self.encoder_t.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.encoder_q.fc = get_mlp(feat_dim, hidden_dim, proj_dim)

        # prediction layer
        # self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)
        self.predict_q = self._build_GCN(GCN_configs_q)
        # self.predict_t = sekf._build_GCN(GCN_configs_t)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, proj_dim))
        # normalize the queue embeddings
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # initialize the labels queue (For Purity measurement)
        self.register_buffer('labels', -1*torch.ones(self.mem_bank_size).long())
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)

    @torch.no_grad()
    def _build_GCN(self, GCN_configs):
        GCNs = MultiLayerGraphNeuralNetwork(GCN_configs)
        return GCNs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels):
        batch_size = targets.shape[0]

        ptr = int(self.queue_ptr)
        assert self.mem_bank_size % batch_size == 0

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets
        self.labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_t, labels):
        # compute query features
        feat_q = self.encoder_q(im_q)

        # compute predictions for instance level regression loss
        # query = self.predict_q(feat_q, torch.eye(len(feat_q), len(feat_q) // torch.cuda.device_count()).cuda())
        bsz = len(im_q)
        adj = torch.eye(self.topk).reshape((1,)+torch.eye(self.topk).shape).repeat(bsz,1,1).cuda()
        query = self.predict_q(feat_q)

        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            current_target = self.encoder_t(im_t)
            current_target = nn.functional.normalize(current_target, dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            self._dequeue_and_enqueue(current_target, labels)

        # calculate mean shift regression loss
        targets = self.queue.clone().detach()
        # calculate distances between vectors
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [query, targets])

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=False)
        nn_dist_q = torch.gather(dist_q, 1, nn_index)

        labels = labels.unsqueeze(1).expand(nn_dist_q.shape[0], nn_dist_q.shape[1])
        labels_queue = self.labels.clone().detach()
        labels_queue = labels_queue.unsqueeze(0).expand((nn_dist_q.shape[0], self.mem_bank_size))
        labels_queue = torch.gather(labels_queue, dim=1, index=nn_index)
        matches = (labels_queue == labels).float()

        loss = (nn_dist_q.sum(dim=1) / self.topk).mean()
        purity = (matches.sum(dim=1) / self.topk).mean()

        return loss, purity


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class TwoCropsTransform:
    """Take two random crops of one image as the query and target."""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        print(self.weak_transform)
        print(self.strong_transform)

    def __call__(self, x):
        q = self.strong_transform(x)
        t = self.weak_transform(x)
        return [q, t]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# Create train loader
def get_train_loader(opt):
    traindir = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_strong = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_weak = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    if opt.dataset == 'imagenet100':
        if opt.weak_strong:
            train_dataset = Image100FolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong))
            )
        else:
            train_dataset = Image100FolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong))
            )
    else:
        if opt.weak_strong:
            train_dataset = ImageFolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong))
            )
        else:
            train_dataset = ImageFolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong))
            )
    print('==> train dataset')
    print(train_dataset)

    # NOTE: remove drop_last
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)

    return train_loader


def main():
    global writer
    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_path, 'events'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.checkpoint_path, 'events'))

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.checkpoint_path, 'logs'),
            filepath=os.path.abspath(__file__)
        )

        def print_pass(*arg):
            logger.info(*arg)
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print(args)

    train_loader = get_train_loader(args)

    mean_shift = MeanShift(
        args.arch,
        m=args.momentum,
        mem_bank_size=args.mem_bank_size,
        topk=args.topk,
        GCN_configs_q=args.GCN_configs_q,
        GCN_configs_t=args.GCN_configs_t
    )
    # mean_shift = torch.nn.DataParallel(mean_shift)
    mean_shift.data_parallel()
    mean_shift = mean_shift.cuda()
    print(mean_shift)

    params = [p for p in mean_shift.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1

    if args.weights:
        print('==> load weights from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt['state_dict']
        msg = mean_shift.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        print(msg)

    if os.path.exists(os.path.join(args.checkpoint_path, 'ckpt_last.pth')) and not args.resume:
        try:
            args.resume = os.path.join(args.checkpoint_path, 'ckpt_last.pth')
            print('==> resume from checkpoint: {}'.format(args.resume))
            ckpt = torch.load(args.resume)
            print('==> resume from epoch: {}'.format(ckpt['epoch']))
            mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
            if not args.restart:
                optimizer.load_state_dict(ckpt['optimizer'])
                args.start_epoch = ckpt['epoch'] + 1
        except:
            import glob
            candidate_resumes = glob.glob(os.path.join(args.checkpoint_path, 'ckpt_epoch_*.pth'))
            epoch_num = [int(x.split('/')[-1].split('ckpt_epoch_')[-1].split('.pth')[0]) for x in candidate_resumes]

            if len(epoch_num) == 0:
                if args.resume:
                    print('==> resume from checkpoint: {}'.format(args.resume))
                    ckpt = torch.load(args.resume)
                    print('==> resume from epoch: {}'.format(ckpt['epoch']))
                    mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
                    if not args.restart:
                        optimizer.load_state_dict(ckpt['optimizer'])
                        args.start_epoch = ckpt['epoch'] + 1
                else:
                    print('==> resume from scratch!')
            else:
                max_epoch_num = max(epoch_num)
                args.resume = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=max_epoch_num))
                print('==> resume from checkpoint: {}'.format(args.resume))
                ckpt = torch.load(args.resume)
                print('==> resume from epoch: {}'.format(ckpt['epoch']))
                mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
                if not args.restart:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    args.start_epoch = ckpt['epoch'] + 1
    elif args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1
    else:
        print('==> resume from scratch!')

    # if args.resume:
    #     print('==> resume from checkpoint: {}'.format(args.resume))
    #     ckpt = torch.load(args.resume)
    #     print('==> resume from epoch: {}'.format(ckpt['epoch']))
    #     mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
    #     if not args.restart:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         args.start_epoch = ckpt['epoch'] + 1

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss_epoch, purity_epoch = train(epoch, train_loader, mean_shift, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # saving metics
        writer.add_scalar('Loss/Epoch', loss_epoch, epoch)
        writer.add_scalar('Purity/Epoch', purity_epoch, epoch)

        # saving the model
        print('==> Saving last...')
        state = {
            'opt': args,
            'state_dict': mean_shift.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        save_file = os.path.join(args.checkpoint_path, 'ckpt_last.pth')
        torch.save(state, save_file)

        # help release GPU memory
        del state
        torch.cuda.empty_cache()


        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': mean_shift.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()

    with open(os.path.join(args.checkpoint_path, 'complete.txt'), 'w') as f:
        f.writelines('complete!')

def train(epoch, train_loader, mean_shift, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    mean_shift.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    purity_meter = AverageMeter()

    end = time.time()
    for idx, (indices, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        loss, purity = mean_shift(im_q=im_q, im_t=im_t, labels=labels)
        loss = loss.mean()
        purity = purity.mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))
        purity_meter.update(purity.item(), im_q.size(0))
        writer.add_scalar('Loss/Iter', loss.item(), idx + epoch * len(train_loader))
        writer.add_scalar('Purity/Iter', purity.item(), idx + epoch * len(train_loader))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'purity {purity.val:.3f} ({purity.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   purity=purity_meter,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg, purity_meter.avg


if __name__ == '__main__':
    main()
