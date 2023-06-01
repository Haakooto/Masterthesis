"""
Rework of model_attack,
"""

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
import pandas as pd
from socket import gethostname


import utils.utils as utils
import utils.fitting as fitting
from utils.argparsing import simple_attack_argparser as saap
from utils.dataloader import load_data
from utils.CustomObjects import DottableDefaultDict
from layers.classification import BioNet, BPNet
from layers.efficient_local_learning import LocalLearning
from layers.attacks import fgsm_Manager, pgd_Manager
from utils.telebot import send_msg

from IPython import embed

#! See explaination in utils/utils.py for these
famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

#! In this file, I do not define my own object to store variables
# ? Instead I use the argparse.Namespace object
# ? The functions that start with 'DO_' add variables to this namespace,
# ? but do not return anything
#! This is similar to the use of ! in Julia-functions


def DO_load_data(args):
    args.data = load_data(args, train=False)


def DO_load_model(args):
    if args.BP:
        args.model = BPNet(path=hebb_path(args.name, args.family))
        args.model.load_model(name=args.name, device=args.device)
    else:
        args.model = BioNet(path=clas_path(
            args.hebb_name, args.name, args.family))
        args.model.load_model(hebb_name=args.hebb_name,
                              name=args.name, device=args.device)

    if not os.path.isdir(f"{args.model.path}/attacks"):
        os.mkdir(f"{args.model.path}/attacks")
    if not os.path.isdir(f"{args.model.path}/attacks/{args.prefix}"):
        os.mkdir(f"{args.model.path}/attacks/{args.prefix}")


def model_attacked(args):
    if os.path.isfile(f"{args.model.path}/attacks/{args.prefix}/rhos.npy"):
        if not args.force:
            args.results = ResultsContainer(args)
            args.results.load()
            return True
    with open(f"{args.model.path}/attacks/{args.prefix}_results.txt", "w") as f:
        f.write("")  # ! Empty the results file
    return False


def DO_initiate_attack(args):
    args.data.reset()
    args.data.to(args.device)
    if args.certainty > 1:
        args.data.keep_top(args.model, args.certainty)
    else:
        args.data.drop_wrong(args.model, certainty=args.certainty)
    args.data.detach()

    args.samples = len(args.data)
    if args.samples < 500:
        return False

    # * Make the batches more even
    args.num_batches = args.samples // args.batch_size + \
        (args.samples % args.batch_size != 0)
    args._batch_size = args.samples // args.num_batches + \
        (args.samples % args.num_batches != 0)
    return True


def DO_make_prefix(args):
    if args.certainty > 1:
        prefix = f"{args.attack}_S{args.certainty}"
    else:
        prefix = f"{args.attack}_C{args.certainty}"
    if args.attack == "pgd":
        prefix += f"_s{args.size:.2}_e{args.step:.2}"
    args.prefix = prefix.replace(".", "")


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes:.0f}m {seconds:.1f}s"
        else:
            hours, minutes = divmod(minutes, 60)
            return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"


class ResultsContainer:
    def __init__(self, args):
        self.args = args
        self.rhos = {i: [] for i in range(args.data.num_classes)}
        self.rhos["all"] = []
        if args.attack == "pgd":
            self.steps = {i: [] for i in range(args.data.num_classes)}
            self.steps["all"] = []

    def save(self):
        np.save(
            f"{self.args.model.path}/attacks/{self.args.prefix}/rhos.npy", self.rhos)
        if self.args.attack == "pgd":
            np.save(
                f"{self.args.model.path}/attacks/{self.args.prefix}/steps.npy", self.steps)

    def load(self):
        self.rhos = np.load(
            f"{self.args.model.path}/attacks/{self.args.prefix}/rhos.npy", allow_pickle=True).item()
        if self.args.attack == "pgd":
            self.steps = np.load(
                f"{self.args.model.path}/attacks/{self.args.prefix}/steps.npy", allow_pickle=True).item()


def DO_attack(args, Manager):
    torch.autograd.set_detect_anomaly(True)  # * Torch recomended me to do this

    args.results = ResultsContainer(args)
    print(f"Attacking {args.attack} with certainty {args.certainty}")
    print(f"samples: {args.samples}")

    loader = tqdm(range(0, args.samples, args._batch_size),
                  desc=f"{args.attack} attack", leave=False)

    for bidx in loader:
        x = args.data.x[bidx: bidx + args._batch_size]
        y = args.data.targets[bidx: bidx + args._batch_size]
        x, y = x.to(args.device), y.to(args.device)

        if not args.BP:
            x = x.flatten(1)

        Adversary = Manager(args, x, y)
        if not Adversary.attack():
            return False
        Adversary.save_metrics()

    loader.close()
    print(
        f"Model attack complete. Time lapsed: {format_duration(loader.format_dict['elapsed'])}")
    args.results.save()
    return True if len(args.results.rhos["all"]) > 100 else False


def DO_estimate_robustness(args):
    if args.plot:
        epses = np.load(f"{args.model.path}/attacks/{args.prefix}/rhos.npy", allow_pickle=True).item()["all"]
        args.results = ResultsContainer(args)
        args.results.rhos["all"] = epses
        args.samples = len(epses)
    else:
        epses = np.asarray(args.results.rhos["all"])
    epses.sort()
    count = np.arange(len(epses))[::-1] / len(epses)

    cutoff = np.argmin(np.abs(count - args.cutoff))  # ! x% cutoff
    epses = epses[:cutoff]
    count = count[:cutoff]

    pad = int(len(epses) * args.pad)  # ! x% padding
    eps_pad = np.linspace(0, epses[0], pad)

    count = np.concatenate((np.ones(pad), count, np.zeros(pad)))
    epses = np.concatenate((eps_pad, epses, epses[-1] + eps_pad))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fitting.generalized_logistic(epses[:, None], count)
        if fit is None:
            return False
        if fit.R2 < 0.95:
            return False

    # #* Unpad data
    # pads = (epses[:pad], epses[-pad:])
    # epses = epses[pad:-pad]
    # count_pad = (count[:pad], count[-pad:])
    # count = count[pad:-pad]

    # * When plotting I don't want to show a very long tail
    # * Find the std of the difference in unpadded epses,
    # * and hide the points above the first eps that is 2 std away
    diffs = epses[pad:] - epses[:-pad]
    std = diffs.std()
    try:
        d = np.where(diffs > 4 * std)[0][0]
    except IndexError:
        d = np.where(diffs > 2 * std)[0][0]

    # * If the first eps is too large, hide the points above the first eps that is 0.2 away
    a = np.where(epses > 0.2)[0]
    if len(a) > 0:
        d = min(d, a[0])

    # #* save pads
    # fit.x0 = pads[0]
    # fit.x1 = pads[1]
    # fit.y0 = count_pad[0]
    # fit.y1 = count_pad[1]

    # * save data
    fit.x = epses[:pad + d]
    fit.y = count[:pad + d]
    fit.eval = fit.eval[:pad + d]
    fit.max = fit.x.max()
    args.fit = fit
    return True


def record_results(args):
    info = {}
    model = args.model
    fit = args.fit

    if not args.BP:
        hebb = model.Hebbian
        info["hebb_name"] = args.hebb_name
        info["name"] = args.name
        info["K"] = hebb.K
        info["width"] = hebb.width
        info["p"] = hebb.p
        info["k"] = hebb.k
        info["delta"] = hebb.delta
        info["hebb_eps0"] = hebb.eps0
        info["hebb_lr_type"] = hebb.lr_type
        info["hebb_epochs"] = hebb.num_epochs
        info["ratio"] = hebb.ratio
        info["clas_eps0"] = model.eps0
        info["clas_epochs"] = model.num_epochs
        info["clas_lr_type"] = model.lr_type
    else:
        info["name"] = args.name
        info["K"] = model.K
        info["width"] = model.width
        info["eps0"] = model.eps0
        info["lr_type"] = model.lr_type
    info["maxpool_kernel"] = model.maxpool_kernel
    info["power"] = model.power
    info["activate"] = model.activate
    info["test_acc"] = model.test_acc
    info["seed"] = args.model.seed

    # * record the attack parameters
    info["attack"] = args.attack
    info["certainty"] = args.certainty
    if args.attack == "pgd":
        info["step_size"] = args.step
        info["proj_size"] = args.size
    info["samples"] = args.samples
    info["cutoff"] = args.cutoff
    info["pad"] = args.pad
    info["crit_eps"] = fit.crit_eps
    info["std_err_eps"] = fit.err
    info["MSE"] = fit.MSE
    info["R2"] = fit.R2
    info.update(fit.popt)

    np.save(f"{args.model.path}/attacks/{args.prefix}/fit.npy", dict(fit))

    output = f"\n{args.attack} attack results for {args.model.hebb_name if not args.BP else ''} {args.model.name} with certainty {args.certainty}%\n"
    output += "=" * len(output) + "\n"
    for key, val in info.items():
        output += f"    {key:<25}: {val}\n"

    og = sys.stdout
    with open(f"{args.model.path}/attacks/{args.prefix}/results.txt", "w") as f:
        sys.stdout = f
        print(output)
    sys.stdout = og

    if not args.fam_attack:
        return

    df = pd.DataFrame([info])
    df.to_csv(args.file_name, mode="a", header=not os.path.isfile(
        args.file_name), index=False)


def plot_results(args):
    # * Plot the range and critical eps
    fig, ax = plt.subplots()

    # ax.plot(args.fit.x, args.fit.y, "bo-", zorder=2, label=f"Measured accuracy")
    # ax.plot(args.fit.x, args.fit.eval, "r--", zorder=2, label=rf"Fit. MSE: {args.fit.MSE:.5f}, $R^2$: {args.fit.R2:.5f}")
    # ax.axvline(x=args.fit.crit_eps, color="k", linestyle="--", zorder=2, label=rf"Crit. eps: {args.fit.crit_eps:.4f}")
    # # ax.plot(args.fit.x0, args.fit.y0, "ko-", alpha=0.7, zorder=1, label=f"Padding for fit")
    # # ax.plot(args.fit.x1, args.fit.y1, "ko-", alpha=0.7, zorder=1, label=f"Padding for fit")

    ax.set_title(f"Robustness of {args.model.hebb_name if not args.BP else ''} {args.model.name} against {args.attack} attack")
    # ax.set_xlabel("Attack strength")
    # ax.set_ylabel("Accuracy")
    # ax.set_xlim(0, args.fit.max)
    # ax.legend()

    ax.plot(args.fit.x, args.fit.y, "bo-", zorder=2,label="Measured accuracy, " + r"$\mathcal{C}\geq$" + f"{args.certainty:.0%}")
    ax.plot(args.fit.x, args.fit.eval, "r--", zorder=2,label=rf"Fit. MSE: {args.fit.MSE:.5f}, $R^2$: {args.fit.R2:.4f}")
    ax.axvline(x=args.fit.crit_eps, color="k", linestyle="--", zorder=2, label=r"$\mathcal{R}$" + f"= {args.fit.R:.1u}")
    ax.axvline(x=args.fit.crit_eps + args.fit.err, color="k", linestyle=":", zorder=2)
    ax.axvline(x=args.fit.crit_eps - args.fit.err, color="k", linestyle=":", zorder=2)

    ax.set_xlabel("Attack strength", fontsize=14)
    ax.set_ylabel("Relative accuracy", fontsize=14)
    ax.set_xlim(0, args.fit.max)
    ax.legend(fontsize=14)

    plt.savefig(f"{args.model.path}/attacks/{args.prefix}/curve.png")
    plt.close()

    return
    # * Plot the results for each class
    fig, ax = plt.subplots()

    x = np.arange(len(args.results.rhos))
    means = np.asarray([np.mean(result) if len(result) >
                       0 else 0 for result in args.results.rhos.values()])
    stds = np.asarray([np.std(result) if len(result) >
                      0 else 0 for result in args.results.rhos.values()])
    names = np.asarray([f"Class {i}" for i in args.results.rhos.keys()])
    names[-1] = "All"

    ax.bar(x, means, yerr=stds, align="center",
           alpha=0.5, ecolor="black", capsize=10)
    ax.axhline(y=args.fit.crit_eps, color="r", linestyle="--",
               label=rf"Crit. eps: {args.fit.crit_eps:.5f}")

    for i, n in enumerate(args.results.rhos.keys()):
        ax.text(i, 0, f"{len(args.results.rhos[n])}", color="blue",
                fontweight="normal", ha="center", va="bottom")

    ax.set_ylabel("Smallest epsilon")
    ax.set_xticks(rotation=45, ticks=x, labels=names, ha="right")
    ax.yaxis.grid(True)
    ax.set_title(
        f"Smallest epsilon for {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack")
    ax.set_ylim(bottom=0)
    plt.savefig(f"{args.model.path}/attacks/{args.prefix}/classbar.png")
    plt.close()

    if args.no_hist:
        return
    mx = np.max(args.results.rhos["all"])
    bins = np.arange(0, mx, args.Manager.min_delta * 1000)
    fig, ax = plt.subplots()
    H = ax.hist([args.results.rhos[c] for c in args.results.rhos.keys(
    ) if c != "all"], bins=bins, label=names[:-1], edgecolor="k", stacked=True)
    ax.set_xlabel(r"$\epsilon_{min}$")
    ax.set_ylabel("count")
    ax.set_title(
        f"Smallest epsilon for {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack")
    ax.legend()
    ax.set_ylim(0, H[0].max())
    ax.set_xlim(0, args.fit.x.max())
    plt.savefig(f"{args.model.path}/attacks/{args.prefix}/hist.png")
    plt.close()


def do_everything(args):
    Manager = fgsm_Manager if args.attack == "fgsm" else pgd_Manager

    try:
        DO_load_model(args)
    except FileNotFoundError:
        print(
            f"Failed loading model {args.model.hebb_name if not args.BP else ''} {args.model.name}")
        return

    print(
        f"Model {args.model.hebb_name if not args.BP else ''} {args.model.name} Loaded")

    redo = model_attacked(args)
    if not redo:
        if not DO_initiate_attack(args):
            args.fail_list["samples"].append(
                (args.model.hebb_name if not args.BP else '', args.model.name, args.samples))
            print("Not enough samples, skipping")
            print()
            return
        if not DO_attack(args, Manager):
            args.fail_list["attack"].append(
                (args.model.hebb_name if not args.BP else '', args.model.name, len(args.results.rhos["all"])))
            print("Attack failed, skipping")
            print()
            return
    else:
        print("Model already attacked, skipping")
    if not DO_estimate_robustness(args):
        args.fail_list["fit"].append(
            (args.model.hebb_name if not args.BP else '', args.model.name, len(args.results.rhos["all"])))
        print("Robustness estimation failed, skipping")
        print()
        return
    record_results(args)
    plot_results(args)
    print()

def only_plot(args):
    print("plotting")
    try:
        DO_load_model(args)
    except FileNotFoundError:
        print(
            f"Failed loading model {args.model.hebb_name if not args.BP else ''} {args.model.name}")
        return

    print(
        f"Model {args.model.hebb_name if not args.BP else ''} {args.model.name} Loaded")
    
    if not os.path.isfile(f"{args.model.path}/attacks/{args.prefix}/rhos.npy"):
        print("No results found, skipping")
        print()
        return

    if not DO_estimate_robustness(args):
        args.fail_list["fit"].append(
            (args.model.hebb_name if not args.BP else '', args.model.name, len(args.results.rhos["all"])))
        print("Robustness estimation failed, skipping")
        print()
        return
    print("Model replotted!")
    record_results(args)
    plot_results(args)
    print()


def main(args=None):
    print()
    if args is None:
        args = saap()
    print(args)
    print(f"Device: {args.device}")

    # ! args.data is now the CustomDatasetClass from dataloader.py
    DO_load_data(args)
    DO_make_prefix(args)
    args.fail_list = {"samples": [], "attack": [], "fit": []}

    if args.fam_attack:
        try:
            args.file_name = f"{hebb_path(fam=args.family)}/AttackData/{args.family}_{args.prefix}_complete_record.csv"
            if not os.path.isdir("/".join(args.file_name.split("/")[:-1])):
                os.mkdir("/".join(args.file_name.split("/")[:-1]))
            if os.path.exists(args.file_name):
                os.rename(args.file_name, args.file_name[:-4] + "_old.csv")
                
            if args.BP:
                for name in existing_models(fam=args.family):
                    args.name = name
                    # if args.name[-2:] not in ["01", "16"]:
                    #     continue
                    if args.plot:
                        only_plot(args)
                    else:
                        do_everything(args)
            else:
                # last_crash = "K20_w12_42_copy0"
                hebbs = existing_models(fam=args.family)
                # hebbs = hebbs[hebbs.index(last_crash)-1:]
                for hebb_name in hebbs:
                    # M = LocalLearning(path=hebb_path(hebb_name, fam=args.family))
                    # M.load_model(name=hebb_name, device=torch.device("cpu"))
                    # if M.ratio < 0.8:
                    #     continue
                    for name in existing_models(name=hebb_name, fam=args.family):
                        args.name = name
                        # if args.name[-2:] not in ["16", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65"]:
                        #     continue
                        args.hebb_name = hebb_name
                        if args.plot:
                            only_plot(args)
                        else:
                            do_everything(args)
            print(args.fail_list)
            send_msg(f"Family attack finished on {gethostname()}")
        except FileNotFoundError:
            send_msg(f"Family attack failed on {gethostname()}")
            raise
    else:
        if args.plot:
            only_plot(args)
        else:
            do_everything(args)
    return  # ! This is the end of the main-function


def missing_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for hebb_name in existing_models(fam="Ensamblers"):
        M = LocalLearning(path=hebb_path(hebb_name, fam="Ensamblers"))
        M.load_model(name=hebb_name, device=device)
        if M.ratio < 0.8:
            print(f"Ratio too low: {hebb_name}")
            continue

        for name in existing_models(name=hebb_name, fam="Ensamblers"):
            try:
                model = BioNet(path=clas_path(hebb_name, name, "Ensamblers"))
                model.load_model(hebb_name=hebb_name, name=name, device=device)
            except:
                print(f"Missing model: {hebb_name} {name}")


if __name__ == "__main__":
    main()
    # missing_models()
