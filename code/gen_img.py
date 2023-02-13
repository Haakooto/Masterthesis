from tqdm import tqdm
import torch
from torch.nn import functional as FF
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import gaussian_filter1d

from utils.dataloader import load_data
from layers.classification import BioNet, BPNet
from utils.utils import path_utils_pack, random_seed_generator, pick_device
from utils.plot import format_nice_plot

famy_path, hebb_path, clas_path, existing_models = path_utils_pack

def main(args=None):
    parser=argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("--name", nargs="+", type=str, required=True)
    add_arg("--family", type=str, default="None")
    add_arg("--dataset", type=str, default="CIFAR")
    add_arg("--seed", type=int, default=-1)
    add_arg("--device", type=str, default="cuda")
    add_arg("--clas", type=int, default=-1)
    add_arg("--iters", type=int, default=100)
    add_arg("--lr", type=float, default=25)
    add_arg("--L2", type=float, default=1e-3)
    add_arg("--blur_every", type=int, default=10)
    add_arg("--blur_sigma", type=float, default=0.5)
    args = parser.parse_args(args)

    args.seed = random_seed_generator(seed=args.seed) if args.seed < 1e4 else args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = pick_device(args.device)
    if args.family == "None":
        args.family = famy_path
    if args.clas == -1:
        args.clas = np.random.randint(0, 10)
    args.clas = torch.tensor((args.clas,)).long().to(args.device)

    if len(args.name) == 1:
        args.name = args.name[0]
        assert args.name in existing_models(fam=args.family)
        model = BPNet(path=hebb_path(args.name, args.family))
        model.load_model(name=args.name, device=args.device)
    else:
        args.hebb_name = args.name[0]
        args.name = args.name[1]
        assert args.hebb_name in existing_models(fam=args.family)
        assert args.name in existing_models(name=args.hebb_name, fam=args.family)
        model = BioNet(path=clas_path(args.hebb_name, args.name, args.family))
        model.load_model(hebb_name=args.hebb_name, name=args.name, device=args.device)
    
    x = load_data(args, train=False)
    args.clas_name = x.classes[args.clas]

    path = f"{model.path}/image_generation"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    run = f"{args.clas_name}_{str(time.time())[-4:]}"
 
    args.model = model
    args.path = path
    args.run = run
    return args

def blur(X, sigma=1):
    X_np = X.detach().cpu().numpy()
    X_np = gaussian_filter1d(X_np, sigma=sigma, axis=1)
    X_np = gaussian_filter1d(X_np, sigma=sigma, axis=2)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def image_generation(args):
    channels = 3
    size = 32
    if args.dataset == "MNIST":
        channels = 1
        size = 28
    img = torch.randn((1, size, size, channels), device=args.device)

    with tqdm(range(args.iters+1)) as pbar:
        for i in pbar:
            ox, oy = np.random.randint(0, 4), np.random.randint(0, 4)
            img = torch.roll(img, shifts=(ox, oy), dims=(1, 2))
            new, loss = update(img, args)
            img.copy_(new.data)
            img = torch.roll(img, shifts=(-ox, -oy), dims=(1, 2))

            if i % args.blur_every == 0:
                img = blur(img, sigma=0.75)

            pbar.set_description(f"{args.run}. log(Loss): {torch.log10(loss):.4f}")

    save(img, i, args)

def update(img, args):
    img = img.clone().detach().requires_grad_(True)
    pred = args.model(img)
    loss = FF.nll_loss(pred, args.clas)
    
    grad = torch.autograd.grad(loss, img)[0]
    grad /= grad.norm()

    with torch.no_grad():
        img -= args.lr * (grad + img * args.L2)

    return img.detach(), loss

def save(img, i, args):
    img = img.detach().cpu().numpy()[0]
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title(f"{args.clas_name} ({args.iters} iters)")
    format_nice_plot()
    plt.savefig(f"{args.path}/{args.run}_i{str(i).zfill(3)}.png")
    plt.clf()
    

if __name__ == "__main__":
    args = main()
    image_generation(args)
