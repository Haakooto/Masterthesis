"""
This file implements different attack-methods
Should not be run directly, rather used by model_attack.py
"""
import torch
from torch.nn import functional as FF

def nothing(model, x, target, args):
    return 0

def random(model, x, target, args):
    return torch.randn_like(x)

def fgsm(model, x, target, args):
    x.requires_grad = True
    output = model(x)
    loss = FF.nll_loss(output, target, reduction='sum')
    loss.backward()
    return x.grad.detach().sign()

# def PGD(model, x, target, args):
    # x.requires_grad = True
    # for _ in range(args.n_steps):
    #     output = model(x)
    #     loss = torch.nn.functional.cross_entropy(output, target)
    #     loss.backward()
    #     x = x.grad.detach().sign()
    #     x.grad.zero_()
    # return x

# def cw(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x.grad.zero_()
#     return x

# def PGD_Linf(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x = torch.min(torch.max(x, x - args.eps), x + args.eps)
#         x.grad.zero_()
#     return x

# def rotate(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x = torch.rot90(x, 1, [2, 3])
#         x.grad.zero_()
#     return x

# def occlude(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x = x * (1 - torch.rand_like(x) < args.occlude_prob)
#         x.grad.zero_()
#     return x

# def deepfool(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x = x * (1 - torch.rand_like(x) < args.occlude_prob)
#         x.grad.zero_()
#     return x

# def jsma(model, x, target, args, strength):
#     x.requires_grad = True
#     for _ in range(args.n_steps):
#         output = model(x)
#         loss = torch.nn.functional.cross_entropy(output, target)
#         loss.backward()
#         x = x + strength * x.grad.detach().sign()
#         x = x * (1 - torch.rand_like(x) < args.occlude_prob)
#         x.grad.zero_()
#     return x
