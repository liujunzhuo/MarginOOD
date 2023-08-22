import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# learnable prototypes. For test only
class Prototype(nn.Module):
    """
    learnable prototypes
    Use EMA
    """

    def __init__(self, args):
        super(Prototype, self).__init__()
        self.args = args
        self.prototypes = nn.Parameter(torch.Tensor(self.args.n_cls, self.args.feat_dim))  # prototypes
        nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

    # def init_class_prototypes(self):
    #     """Initialize class prototypes"""
    #     self.model.eval()
    #     start = time.time()
    #     prototype_counts = torch.zeros(self.args.n_cls, dtype=torch.float32).cuda()
    #     prototypes = torch.zeros(self.args.n_cls, self.args.feat_dim, dtype=torch.float32).cuda()
    #     with torch.no_grad():
    #         for input, target in self.loader:
    #             input, target = input.cuda(), target.cuda()
    #             features = self.model(input)  # extract normalized features
    #             prototype_counts.index_add_(0, target, torch.ones_like(target, dtype=torch.float32).cuda())
    #             prototypes.index_add_(0, target, features)
    #         prototypes /= prototype_counts.unsqueeze(1)
    #         prototypes = F.normalize(prototypes, dim=1)
    #     self.prototypes = torch.nn.Parameter(prototypes.detach())
    #     duration = time.time() - start
    #     print(f'Time to initialize prototypes: {duration:.3f}')
