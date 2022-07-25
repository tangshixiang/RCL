import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import torch.nn.functional as F
from util import subset_classes

from tools import *
import models.resnet as resnet
# from models.alexnet import AlexNet
# from models.mobilenet import MobileNetV2

import cv2
from PIL import Image

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

parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet100'],
                    help='use full or subset of the dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/distill_1', type=str,
                    help='experiment output directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                    help='lr drop schedule')
parser.add_argument('--restart', action='store_true')

best_acc1 = 0


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)

import pdb


def load_weights(model, wts_path):
    wts = torch.load(wts_path)
    # pdb.set_trace()
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts

    ckpt = {k.replace('encoder_t.module.', ''): v for k, v in ckpt.items()}
    ckpt = {k.replace('module.encoder_t.', ''): v for k, v in ckpt.items()}

    state_dict = {}
    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    model.load_state_dict(state_dict)
    print(model)


def get_model(arch, wts_path):
    if arch == 'alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif arch == 'pt_alexnet':
        model = models.alexnet()
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        load_weights(model, wts_path)
    elif arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif 'resnet' in arch:
        # model = models.__dict__[arch]()
        model = resnet.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


class MemoryCacheImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(MemoryCacheImageFolder, self).__init__(*args, **kwargs)
        self.initialized = False
        self.metas = [x[1] for x in self.samples]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'

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

        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MemoryCacheImageFolderImageNet100(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        self.class_list = '/mnt/lustre/share/tangshixiang/data/ImageNet/ImageNet100_class_map.txt'
        super(MemoryCacheImageFolderImageNet100, self).__init__(*args, **kwargs)
        self.initialized = False
        self.metas = [x[1] for x in self.samples]
        self.num_classes = np.max(self.metas) + 1
        assert self.num_classes == len(np.unique(self.metas)), 'Some labels skipped, please reorganize the labels!'

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

        # sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



def main_worker(args):
    global best_acc1

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


    if args.dataset == 'imagenet':
        train_dataset = MemoryCacheImageFolder(traindir, train_transform)
        val_dataset = MemoryCacheImageFolder(valdir, val_transform)
        train_val_dataset = MemoryCacheImageFolder(traindir, val_transform)
    elif args.dataset == 'imagenet100':
        train_dataset = MemoryCacheImageFolderImageNet100(traindir, train_transform)
        val_dataset = MemoryCacheImageFolderImageNet100(valdir, val_transform)
        train_val_dataset = MemoryCacheImageFolderImageNet100(traindir, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True,
    )

    backbone = get_model(args.arch, args.weights)
    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()


    cached_feats = '%s/var_mean.pth.tar' % args.save
    if not os.path.exists(cached_feats):
        train_feats, _ = get_feats(train_val_loader, backbone, args)
        train_var, train_mean = torch.var_mean(train_feats, dim=0)
        torch.save((train_var, train_mean), cached_feats)
    else:
        train_var, train_mean = torch.load(cached_feats)

    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(args.arch), len(train_dataset.classes)),
    )
    linear = linear.cuda()

    optimizer = torch.optim.SGD(linear.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         linear.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if os.path.exists(os.path.join(args.save, 'ckpt_last.pth')) and not args.resume:
        try:
            args.resume = os.path.join(args.save, 'ckpt_last.pth')
            logger.info('==> resume from checkpoint: {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            logger.info('==> resume from epoch: {}'.format(ckpt['epoch']))
            linear.load_state_dict(checkpoint['state_dict'])
            if not args.restart:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch']
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except:
            import glob
            candidate_resumes = glob.glob(os.path.join(args.save, 'ckpt_epoch_*.pth'))
            epoch_num = [int(x.split('/')[-1].split('ckpt_epoch_')[-1].split('.pth')[0]) for x in candidate_resumes]

            if len(epoch_num) == 0:
                if args.resume:
                    logger.info('==> resume from checkpoint: {}'.format(args.resume))
                    checkpoint = torch.load(args.resume)
                    logger.info('==> resume from epoch: {}'.format(ckpt['epoch']))
                    linear.load_state_dict(checkpoint['state_dict'])
                    if not args.restart:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        args.start_epoch = checkpoint['epoch']
                        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                else:
                    logger.info('==> resume from scratch!')
            else:
                max_epoch_num = max(epoch_num)
                args.resume = os.path.join(args.save, 'ckpt_epoch_{epoch}.pth'.format(epoch=max_epoch_num))
                logger.info('==> resume from checkpoint: {}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                logger.info('==> resume from epoch: {}'.format(ckpt['epoch']))
                linear.load_state_dict(checkpoint['state_dict'])
                if not args.restart:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    args.start_epoch = checkpoint['epoch']
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    elif args.resume:
        logger.info('==> resume from checkpoint: {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        logger.info('==> resume from epoch: {}'.format(ckpt['epoch']))
        linear.load_state_dict(checkpoint['state_dict'])
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = checkpoint['epoch']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        logger.info('==> resume from scratch!')


    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, backbone, linear, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()
        # logger.info('LR: {:f}'.format(lr_scheduler.get_last_lr()[-1]))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, is_best, args.save)

        save_each_checkpoint({
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, epoch + 1, args.save)


    with open(os.path.join(args.save, 'complete.txt'), 'w') as f:
        f.writelines('complete!')


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()
