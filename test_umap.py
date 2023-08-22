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
from utils import plot_utils
from utils.detection_util import set_ood_loader_ImageNet, obtain_feature_from_loader, set_ood_loader_small, \
    get_and_print_results
from utils.util import set_loader_ImageNet, set_loader_small, set_model
from utils.display_results import plot_distribution, print_measures, save_as_dataframe


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
    parser.add_argument('--model', default='resnet34', type=str, help='model architecture')
    parser.add_argument('--embedding_dim', default=512, type=int, help='encoder feature dim')
    parser.add_argument('--feat_dim', default=128, type=int, help='head feature dim')
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


def set_up(args):
    args.log_directory = f"results/{args.in_dataset}/{args.name}/{args.loss}/plot"
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
    if args.in_dataset == 'ImageNet-100':
        train_loader, test_loader = set_loader_ImageNet(args, eval=True)
    else:
        train_loader, test_loader = set_loader_small(args, eval=True)
    try:
        pretrained_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
    except:
        print("loading model as SupCE format")
        pretrained_dict = torch.load(args.ckpt, map_location='cpu')['model']

    net = set_model(args)
    net.load_state_dict(pretrained_dict)
    net.eval()
    return train_loader, test_loader, net


def main(args):
    train_loader, test_loader, net = set_up(args)
    ood_num_examples = len(test_loader.dataset)
    num_batches = ood_num_examples // args.batch_size

    if args.n_cls == 10:
        id_feat, in_label = obtain_feature_from_loader(args, net, test_loader, num_batches, full=True)
    else:
        id_feat = obtain_feature_from_loader(args, net, test_loader, num_batches)
        in_label = ['id' for i in range(id_feat.shape[0])]

    print('preprocessing ID finished')
    if args.in_dataset == 'ImageNet-100':
        out_datasets = ['SUN', 'places365', 'dtd', 'iNaturalist']
    else:
        out_datasets = ['SVHN', 'places365', 'iSUN', 'dtd', 'LSUN', 'LSUN_resize']

    for out_dataset in out_datasets:
        print(f"Evaluting OOD dataset {out_dataset}")
        if args.in_dataset == 'ImageNet-100':
            ood_loader = set_ood_loader_ImageNet(args, out_dataset)
        else:
            ood_loader = set_ood_loader_small(args, out_dataset)

        ood_feat = obtain_feature_from_loader(args, net, ood_loader, num_batches)

        # if args.n_cls != 10:
        #     in_label = ['id' for i in range(id_feat.shape[0])]
        out_label = ["ood" for i in range(ood_feat.shape[0])]
        data = np.concatenate([id_feat, ood_feat])
        label = np.concatenate([in_label, out_label])

        reducer = umap.UMAP(metric="cosine")
        embedding_train = reducer.fit_transform(data)

        plot_utils.plot(embedding_train, label)
        plt.savefig(
            os.path.join(args.log_directory, "ID_{}_OD_{}_{}.svg".format(args.in_dataset, out_dataset, args.mark)))
        plt.show()

    print("Finished")


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    # prform OOD detection
    main(args)
