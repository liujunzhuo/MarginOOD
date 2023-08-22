import os
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import faiss
import umap
from matplotlib import pyplot as plt
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from models.resnet import *
from utils import plot_utils, DisLPLoss
from utils.detection_util import set_ood_loader_ImageNet, obtain_feature_from_loader, set_ood_loader_small, \
    get_and_print_results
from utils.util import set_loader_ImageNet, set_loader_small, set_model, AverageMeter
from utils.display_results import plot_distribution, print_measures, save_as_dataframe
from utils.val import val


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates OOD Detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
    parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--epoch', default="500", type=str, help='which epoch to test')
    parser.add_argument('--gpu', default=4, type=int, help='which GPU to use')

    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--id_loc', default="datasets/CIFAR100", type=str, help='location of in-distribution dataset')
    parser.add_argument('--ood_loc', default="datasets/small_OOD_dataset", type=str, help='location of ood datasets')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--score', default='maha', type=str, help='score options: knn|maha')
    parser.add_argument('--K', default=300, type=int, help='K in KNN score')
    parser.add_argument('--subset', default=False, type=bool, help='whether to use subset for KNN')
    parser.add_argument('--multiplier', default=1, type=float,
                        help='norm multipler to help solve numerical issues with precision matrix')
    parser.add_argument('--model', default='resnet34', type=str, help='model architecture')
    parser.add_argument('--embedding_dim', default=512, type=int, help='encoder feature dim')
    parser.add_argument('--feat_dim', default=128, type=int, help='head feature dim')
    parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
    parser.add_argument('--out_as_pos', action='store_true', help='if OOD data defined as positive class.')
    parser.add_argument('--T', default=1000, type=float, help='temperature: energy|Odin')
    parser.add_argument('--mark', default='default', type=str)

    parser.add_argument('--train_epoch', default="10", type=int)
    parser.add_argument('--print_freq', default="20", type=int)
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    # optimization
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    args = parser.parse_args()
    print(args)

    args.ckpt = f"checkpoints/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"

    if args.in_dataset == "CIFAR-10":
        args.n_cls = 10
    elif args.in_dataset in ["CIFAR-100", 'ImageNet-100']:
        args.n_cls = 100

    return args


def set_up(args):
    args.log_directory = f"results/{args.in_dataset}/{args.name}/{args.loss}/accuracy"
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    train_loader, test_loader = set_loader_small(args, eval=True)

    load_dict = torch.load(args.ckpt)
    pretrained_dict = load_dict['state_dict']

    net_o = SupCEHeadResNet(args)
    net_o.load_state_dict(pretrained_dict)
    net_encoder_state_dict = net_o.encoder.state_dict()

    net = SupCEResNet(name=args.model, num_classes=args.n_cls)
    net.encoder.load_state_dict(net_encoder_state_dict)
    net = net.cuda()
    return train_loader, test_loader, net


def main(args):
    best_acc = 0
    train_loader, test_loader, model = set_up(args)

    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.train_epoch + 1):
        time1 = time.time()
        loss, acc = train(train_loader, model, criterion,
                          optimizer, epoch, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        scheduler.step()

        # eval for one epoch
        loss, val_acc = validate(test_loader, model, criterion, args)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
    with open(os.path.join(args.log_directory, 'accuracy.txt'), "w") as f:
        f.write(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels)
        top1.update(acc1, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels)
            top1.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        res = 100 * correct / total

    return res


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    # prform OOD detection
    main(args)
