import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from einops import rearrange

from utils.dataloader import load_data
from layers.classification import BioNet, BPNet
from utils.utils import path_utils_pack, random_seed_generator, pick_device

famy_path, hebb_path, clas_path, existing_models = path_utils_pack
colorwheel = ['#762a83','#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b','#1b7837']

def parse(args=None):
    parser=argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("--name", nargs="+", type=str, required=True)
    add_arg("--family", type=str, default="None")
    add_arg("--dataset", type=str, default="CIFAR")
    add_arg("--batch_size", type=int, default=1000)
    add_arg("--seed", type=int, default=1)
    add_arg("--device", type=str, default="cuda")
    args = parser.parse_args(args)

    args.seed = random_seed_generator(seed=args.seed) if args.seed < 1e4 else args.seed
    args.device = pick_device(args.device)
    if args.family == "None":
        args.family = famy_path

    assert len(args.name) == 3, "Name must be of the form: [hebb_name] [clas_name] [BP_name]"
    args.hebb_name, args.name, args.bp_name = args.name
    assert args.hebb_name in existing_models(fam=args.family)
    assert args.name in existing_models(name=args.hebb_name, fam=args.family)
    assert args.bp_name in existing_models(fam=args.family)

    if args.BP:
        args.name = args.name[0]
        model = BPNet(path=hebb_path(args.name, args.family))
        model.load_model(name=args.name, device=args.device)
    else:
        args.hebb_name = args.name[0]
        args.name = args.name[1]
        model = BioNet(path=clas_path(args.hebb_name, args.name, args.family))
        model.load_model(hebb_name=args.hebb_name, name=args.name, device=args.device)


    return args


def cov_spectrum(inp_data, args, key, model=None):
    print(key)
    with torch.no_grad():
        device = args.device
        dtype = inp_data.dtype
        if model is None:
            signal = lambda x: rearrange(x.to(device), "n c h w -> n (c h w)")
        else:
            signal = lambda x: rearrange(model.hidden_layer(x.to(device)), "n H P -> (n P) H")

        comps = signal(inp_data[:1]).size(1)

        auto_corr = torch.zeros((comps, comps), device=device, dtype=dtype)
        mean = torch.zeros((comps,), device=device, dtype=dtype)
        pop_size = 0

        K = inp_data[0].mean()

        for bidx in range(0, len(inp_data), args.batch_size):
            x = inp_data[bidx : bidx + args.batch_size]
            if x.size(0) == 0:
                continue
            pop_size += x.size(0)

            pred = signal(x)
            data = pred - K
            auto_corr += data.t() @ data
            mean += data.sum(dim=0)

        cov = auto_corr - mean[None].T @ mean[None] / pop_size
        cov /= pop_size - 1

        evals = torch.linalg.eigvalsh(cov)

        real = torch.sort(evals, descending=True)[0].detach().cpu().numpy()

        args.spectra[key] = real


def gather_the_spectra(args):
    torch.manual_seed(args.seed)
    #* Load models
    bp_model = BPNet(path=hebb_path(args.bp_name, args.family))
    bp_model.load_model(name=args.bp_name, device=args.device)

    ll_model = BioNet(path=clas_path(args.hebb_name, args.name, args.family))
    ll_model.load_model(hebb_name=args.hebb_name, name=args.name, device=args.device)

    #* Load data
    img_data = load_data(args, train=False)
    img_data.to(args.device)

    gauss_data = torch.randn(size=img_data.x.size())

    args.spectra = {}

    cov_spectrum(gauss_data, args, "gauss_data")
    cov_spectrum(gauss_data, args, "gauss_rep_bp", bp_model)
    cov_spectrum(gauss_data, args, "gauss_rep_ll", ll_model)
    cov_spectrum(img_data.x, args, "image_data")
    cov_spectrum(img_data.x, args, "image_rep_bp", bp_model)
    cov_spectrum(img_data.x, args, "image_rep_ll", ll_model)
    ll_model.reset_model()
    cov_spectrum(gauss_data, args, "gauss_rep_virgin", ll_model)
    cov_spectrum(img_data.x, args, "image_rep_virgin", ll_model)

    plot(args)

def plot(args):
    plt.style.use(['seaborn-paper'])
    fig, axs = plt.subplots(1, 2)

    get = lambda x: args.spectra[x]
    aranged = lambda x: np.arange(1, len(get(x)) + 1)
    n1 = lambda x: (aranged(x), get(x)[0] / aranged(x))
    spec = lambda x: (aranged(x), get(x))

    axs[0].set_title("Data")
    axs[0].loglog(*spec("image_data"), color=colorwheel[-2], label="CIFAR10")
    axs[0].loglog(*n1("image_data"),   color="grey",         linestyle="dotted")

    axs[0].loglog(*spec("gauss_data"), color=colorwheel[1], label="Gaussian")
    axs[0].loglog(*n1("gauss_data"),   color="grey",        linestyle="dotted")

    axs[0].legend()
    axs[0].set_xlabel(r"$n$")
    axs[0].set_ylabel(r"$\lambda_n$")

    axs[1].set_title("Representations")
    axs[1].loglog(*spec("image_rep_ll"),     color=colorwheel[-2])
    axs[1].loglog(*n1("image_rep_ll"),       color="grey",        linestyle="dotted")
    axs[1].loglog(*spec("image_rep_bp"), color=colorwheel[-2],linestyle="dashed")
    axs[1].loglog(*n1("image_rep_bp"),   color="grey",        linestyle="dotted")

    axs[1].loglog(*spec("gauss_rep_ll"),     color=colorwheel[1])
    axs[1].loglog(*n1("gauss_rep_ll"),       color="grey",        linestyle="dotted")
    axs[1].loglog(*spec("gauss_rep_bp"), color=colorwheel[1], linestyle="dashed")
    axs[1].loglog(*n1("gauss_rep_bp"),   color="grey",        linestyle="dotted")

    custom_lines = [
    Line2D([0], [0], linestyle="solid", color="black"),
    Line2D([0], [0], linestyle="dashed", color="black"),
    ]
    axs[1].legend(custom_lines, [r"LL", r"BP"])
    axs[1].set_xlabel(r"$n$")
    axs[1].set_ylabel(r"$\lambda_{n}$")

    # ylim = axs[0].get_ylim()
    # axs[1].set_ylim(ylim)

    fig.savefig("figures/spectra.png")
    plt.clf()


if __name__ == "__main__":
    gather_the_spectra(parse())
