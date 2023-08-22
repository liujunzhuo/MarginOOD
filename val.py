import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import faiss
import umap
from matplotlib import pyplot as plt
from torch.backends import cudnn
from tqdm import tqdm

from models.resnet import *
from utils import plot_utils, DisLPLoss
from utils.detection_util import set_ood_loader_ImageNet, obtain_feature_from_loader, set_ood_loader_small, \
    get_and_print_results
from utils.util import set_loader_ImageNet, set_loader_small, set_model
from utils.display_results import plot_distribution, print_measures, save_as_dataframe
from utils.val import val


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates OOD Detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
    parser.add_argument('-b', '--batch-size', default=512, type=int, help='mini-batch size')
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
    args = parser.parse_args()
    print(args)

    args.ckpt = f"checkpoints/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"

    if args.in_dataset == "CIFAR-10":
        args.n_cls = 10
    elif args.in_dataset in ["CIFAR-100", 'ImageNet-100']:
        args.n_cls = 100

    return args


def get_features(args, net, train_loader, test_loader):
    feat_dir = f"feat/{args.in_dataset}/{args.name}/{args.epoch}"
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
        ftrain = obtain_feature_from_loader(args, net, train_loader, num_batches=None, embedding_dim=128)
        with open(f'{feat_dir}/feat.npy', 'wb') as f:
            np.save(f, ftrain)
    else:
        with open(f'{feat_dir}/feat.npy', 'rb') as f:
            ftrain = np.load(f)
    ftest = obtain_feature_from_loader(args, net, test_loader, num_batches=None, embedding_dim=128)
    return ftrain, ftest


def get_mean_prec(args, net, train_loader):
    '''
    used for Mahalanobis score. Calculate class-wise mean and inverse covariance matrix
    '''
    save_dir = os.path.join('feat', f"{args.in_dataset}", f"{args.name}", 'maha')
    mean_loc = os.path.join(save_dir, f'{args.loss}_classwise_mean.pt')
    prec_loc = os.path.join(save_dir, f'{args.loss}_precision.pt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(mean_loc) and os.path.exists(mean_loc):
        classwise_mean = torch.load(mean_loc, map_location='cpu').cuda()
        precision = torch.load(prec_loc, map_location='cpu').cuda()
    else:
        classwise_mean = torch.empty(args.n_cls, args.embedding_dim, device='cuda')
        all_features = torch.zeros((0, args.embedding_dim), device='cuda')
        classwise_idx = {}
        with torch.no_grad():
            for idx, (image, labels) in enumerate(tqdm(train_loader)):
                out_feature = net.intermediate_forward(image.cuda())

                all_features = torch.cat((all_features, out_feature), dim=0)

        targets = np.array(train_loader.dataset.targets)
        for class_id in range(args.n_cls):
            classwise_idx[class_id] = np.where(targets == class_id)[0]

        for cls in range(args.n_cls):
            classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim=0)

        cov = torch.cov(all_features.T.double())
        precision = torch.linalg.pinv(cov).float()
        print(f'cond number: {torch.linalg.cond(precision)}')
        torch.save(classwise_mean, mean_loc)
        torch.save(precision, prec_loc)
    return classwise_mean, precision


def set_up(args):
    # args.log_directory = f"results/{args.in_dataset}/{args.name}/{args.loss}/accuracy"
    # if not os.path.exists(args.log_directory):
    #     os.makedirs(args.log_directory)
    if args.in_dataset == 'ImageNet-100':
        train_loader, test_loader = set_loader_ImageNet(args, eval=True)
    else:
        train_loader, test_loader = set_loader_small(args, eval=True)
    try:
        load_dict = torch.load(args.ckpt)
        pretrained_dict = load_dict['state_dict']
        prototype_dict = load_dict['dis_state_dict']['prototypes']
    except:
        print("loading model as SupCE format")
        pretrained_dict = torch.load(args.ckpt, map_location='cpu')['model']

    # criterion_dis = DisLPLoss(args, net, val_loader, temperature=args.temp).cuda()  # V1: learnable prototypes
    prototypes = nn.Parameter(torch.Tensor(args.n_cls, args.feat_dim)).cuda()
    prototypes.data = prototype_dict
    net = set_model(args)
    net.load_state_dict(pretrained_dict)
    net = net.cuda()
    net.eval()
    return train_loader, test_loader, net, prototypes


def main(args):
    train_loader, test_loader, net, prototypes = set_up(args)

    print('preprocessing ID finished')
    if args.in_dataset == 'ImageNet-100':
        out_datasets = ['SUN', 'places365', 'dtd', 'iNaturalist']
    else:
        out_datasets = ['SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize']

    accuracy = val(net, prototypes, test_loader, args)

    print(accuracy)


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    # prform OOD detection
    main(args)
