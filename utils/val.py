import torch
import torch.nn.functional as F


def val(net, prototypes, test_loader, args):
    net.eval()
    device = torch.device('cuda:0')
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, examples in enumerate(test_loader):
            data, target = examples[0], examples[1]
            data = data.to(device)
            target = target.to(device)
            outputs = net(data)
            prototypes = F.normalize(prototypes, dim=1)
            logits = torch.matmul(outputs, prototypes.T)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy
