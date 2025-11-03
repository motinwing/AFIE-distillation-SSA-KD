import torch
import torch.nn as nn
import math


def norm_prune(norm2dlayer, index):
    norm2dlayer.weight = nn.Parameter(
        torch.index_select(norm2dlayer.weight, dim=0, index=index))
    norm2dlayer.bias = nn.Parameter(
        torch.index_select(norm2dlayer.bias, dim=0, index=index))
    norm2dlayer.register_buffer('running_mean', torch.index_select(norm2dlayer.running_mean, dim=0, index=index))
    norm2dlayer.register_buffer('running_var', torch.index_select(norm2dlayer.running_var, dim=0, index=index))


def conv_prune(convlayer, index_c=None, index_r=None, ifprint=True):
    temp_size = convlayer.weight.size(0)
    if index_c is not None:
        convlayer.weight = nn.Parameter(torch.index_select(convlayer.weight, dim=0, index=index_c))
        if convlayer.bias is not None:
            convlayer.bias = nn.Parameter(torch.index_select(convlayer.bias, dim=0, index=index_c))
    # temp_grad = torch.index_select(convlayer.weight.grad, dim=1, index=wl[i].squeeze())
    if index_r is not None:
        convlayer.weight = nn.Parameter(torch.index_select(convlayer.weight, dim=1, index=index_r))
    # convlayer.weight.grad = temp_grad
    if ifprint:
        print("{0}->{1}\t{2:.3f}".format(temp_size, convlayer.weight.size(0), 1 - convlayer.weight.size(0) / temp_size))


def fc_prune(fclayer, index_c=None, index_r=None):
    if index_c is not None:
        fclayer.weight = nn.Parameter(torch.index_select(fclayer.weight, dim=0, index=index_c))
        if fclayer.bias is not None:
            fclayer.bias = nn.Parameter(torch.index_select(fclayer.bias, dim=0, index=index_c))
    if index_r is not None:
        fclayer.weight = nn.Parameter(torch.index_select(fclayer.weight, dim=1, index=index_r))
    print(fclayer.weight.size(0))


# def conv_criteria(convlayer, criteria_list, theta):
#     unsorted = taylor_criteria(convlayer, theta)
#     return torch.cat((criteria_list, unsorted), dim=0)

def conv_criteria(convlayer, criteria_list, theta):
    unsorted = AFIE_criteria(convlayer)
    return torch.cat((criteria_list, unsorted), dim=0)


def taylor_index(convlayer, threshold, theta):
    w = taylor_criteria(convlayer, theta)
    w = torch.nonzero(w >= threshold, as_tuple=False).squeeze()
    return w


def conv_index(convlayer, threshold, theta):
    w = AFIE_criteria(convlayer)
    w1 = taylor_criteria(convlayer, theta)
    w = threshold / w[0]
    if w >= 1:
        w = 0.9
    sorted = torch.sort(w1, dim=0, descending=False)[0]
    threshold = sorted[int(sorted.size(0) * w)]
    w = taylor_index(convlayer, threshold, theta)
    return w


def taylor_criteria(convlayer, theta):
    w = convlayer.weight.clone().detach()
    k = 2
    w2 = torch.sum(torch.abs(w ** k), dim=(1, 2, 3)).pow(k).view(-1)
    w = w * convlayer.weight.grad
    w = torch.sum(torch.abs(w), dim=(1, 2, 3)).view(-1)
    w = theta * w / torch.norm(w, 2) + (1 - theta) * w2 / torch.norm(w2, 2)
    return w


# def AFIE_criteria(convlayer):
#     w = convlayer.weight.clone().detach()
#     w = torch.sum(w, dim=(2, 3))
#     p = w.size(dim=0)
#     _, s, _ = torch.svd(w)
#     q = s.size(dim=0)
#     s = s*s
#     # s = (s - torch.min(s))/(torch.max(s) - torch.min(s))
#     s = (s+1e-5) / torch.sum(s, dim=0)
#     # print('\n')
#     # print(s)
#     # print('\n')
#     # s = torch.softmax(s, dim=0)
#     # s = s / torch.sum(s, dim=0)
#     w = torch.Tensor([torch.sum(-s * torch.log(s)/math.log(q*0.8), dim=0), p]).to(convlayer.weight.device)
#     return w


def AFIE_criteria(convlayer):
    w = convlayer.weight.clone().detach()
    w = torch.sum(w, dim=(2, 3))
    p = w.size(dim=0)

    x = math.exp(1) + min(list(w.shape)) - 1
    x = math.log(x) - math.exp(1) / x
    w = torch.Tensor([x / p, p]).to(convlayer.weight.device)

    return w
