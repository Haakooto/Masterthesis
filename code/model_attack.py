"""
File for a comprehensive attack on a single model.
Will first do a smallest-attack, and use the result to do a ranged attack,
        in order to automatically find the params needed
        for a good estimation of the critical epsilon
"""

import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

import utils.plot as plot
import utils.utils as utils
import utils.fitting as fitting
from utils.argparsing import attack_argparser
from utils.dataloader import load_data
from utils.CustomObjects import DottableDefaultDict
from layers.classification import BioNet, BPNet
import layers.attacks as attacks
import train_models

from IPython import embed

#! See explaination in utils/utils.py for these
famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack
attack_funcs = utils.get_directly_implemented_funcs(attacks)


class Attack_manager:
    def __init__(self, args, show_pbar=True):
        self.__dict__.update(args.__dict__)
        self.attack_func = attack_funcs[self.attack]
        self.data = load_data(self, train=False)
        self.show_pbar = show_pbar

    def load_model(self):
        if self.BP:
            self.model = BPNet(path=hebb_path(self.name, self.family))
            self.model.load_model(name=self.name, device=self.device)
        else:
            self.model = BioNet(path=clas_path(self.hebb_name, self.name, self.family))
            self.model.load_model(hebb_name=self.hebb_name, name=self.name, device=self.device)

        if not os.path.isdir(f"{self.model.path}/attacks"):
            os.mkdir(f"{self.model.path}/attacks")

        self.data.reset()
        self.data.to(self.device)

    def set_names(self, hebb, name=None):
        if hebb is not None:
            if name is None:
                self.name = hebb
            else:
                self.hebb_name = hebb
                self.name = name

    def do_attack(self, hebb=None, name=None, plot=True):
        self.set_names(hebb, name)
        self.load_model()

        if not self.rerun:
            if os.path.isfile(f"{self.model.path}/attacks/{self.prefix}_results.txt"):
                if self.family_attack:
                    return False
                if input("Already done, do you want to rerun? (y/n) ").lower() != "y":
                    return False
                # print(f"Already done, skipping {self.prefix} for {'' if hebb is None else hebb} {self.name}")
                # return False

        self.data.drop_wrong(self.model, self.batch_size, self.certainty)
        self.data.shuffle()
        self.samples = len(self.data)

        if self.samples <= 500:
            return False

        #* Make the batches more even
        self.num_batches = self.samples // self.batch_size + (self.samples % self.batch_size != 0)
        batch_size = self.samples // self.num_batches + (self.samples % self.num_batches != 0)

        torch.autograd.set_detect_anomaly(True) # * Torch recomended me to do this

        if not self._attack(batch_size):
            return False
        if not self.estimate_critical_eps():
            return False

        if plot:
            self.plot_results()
        self.save_results()
        return True

    def _attack(self, batch_size):
        results = {i: [] for i in range(self.data.num_classes)}
        results["all"] = []

        pbar = range(0, self.samples, batch_size)
        if self.show_pbar:
            pbar = tqdm(pbar, desc=f"{self.attack} attack on {self.samples} samples")

        for bidx in pbar:
            x = self.data.x[bidx: bidx + batch_size]
            y = self.data.targets[bidx: bidx + batch_size]
            Eps = self.eps.reset(len(x), self.device)

            dx = self.attack_func(self.model, x, y, self)

            while not Eps.converged.all():
                Px = x + dx * Eps.strength.reshape(-1, 1, 1, 1)
                clamped = Px.clamp(min=0, max=1)

                with torch.no_grad():
                    pred = self.model(clamped).argmax(dim=1)
                    correct = pred.eq(y)
                    Eps.update(correct)

            for b in range(len(x)):
                results["all"].append(Eps.strength[b].item())
                if Eps.strength[b] > 1:
                    continue
                results[y[b].item()].append(Eps.strength[b].item())

        self.results = results
        if self.show_pbar:
            pbar.close()
        return True

    def estimate_critical_eps(self):
        epses = np.asarray(self.results["all"])
        epses.sort()
        count = np.arange(len(epses))[::-1] / len(epses)

        cutoff = np.argmin(np.abs(count - self.cutoff))  #! x% cutoff
        epses = epses[:cutoff]
        count = count[:cutoff]

        pad = int(len(epses) * self.pad)  #! x% padding
        eps_pad = np.linspace(0, epses[0], pad)

        count = np.concatenate((np.ones(pad), count, np.zeros(pad)))
        epses = np.concatenate((eps_pad, epses, epses[-1] + eps_pad))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fitting.generalized_logistic(epses[:, None], count)
            if fit is None:
                return False

        s = np.where(epses < 1)
        fit.x = epses[s]
        fit.y = count[s]
        fit.eval = fit.eval[s]
        self.fit = fit
        return True
    
    @property
    def prefix(self):
        return f"{self.attack}_{str(self.certainty).replace('.', '')}"

    def plot_results(self):
        #* Plot the range and critical eps
        fig, ax = plt.subplots()

        ax.plot(self.fit.x, self.fit.y, "bo-", label=f"Measured, inital certainty: {self.certainty}")
        ax.plot(self.fit.x, self.fit.eval, "r--", label=rf"Logistic fit. MSE: {self.fit.MSE:.5f}, $R^2$: {self.fit.R2:.5f}")
        ax.axvline(x=self.fit.crit_eps, color="k", linestyle="--", label=rf"Crit. eps: {self.fit.crit_eps:.5f}")

        ax.set_title(f"Accuracy of {self.model.hebb_name if not self.BP else ''} {self.model.name} on {self.dataset} for {self.attack} attack")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, self.fit.x.max())
        ax.legend()

        plt.savefig(f"{self.model.path}/attacks/{self.prefix}_eps_curve.png")
        plt.close()

        #* Plot the results for each class
        fig, ax = plt.subplots()

        x = np.arange(len(self.results))
        means = np.asarray([np.mean(result) if len(result) > 0 else 0 for result in self.results.values()])
        stds = np.asarray([np.std(result) if len(result) > 0 else 0 for result in self.results.values()])
        names = np.asarray([f"Class {i}" for i in self.results.keys()])
        names[-1] = "All"

        ax.bar(x, means, yerr=stds, align="center", alpha=0.5, ecolor="black", capsize=10)
        ax.axhline(y=self.fit.crit_eps, color="r", linestyle="--", label=rf"Crit. eps: {self.fit.crit_eps:.5f}")

        for i, n in enumerate(self.results.keys()):
            ax.text(i, 0, f"{len(self.results[n])}", color="blue", fontweight="normal", ha="center", va="bottom")

        ax.set_ylabel("Smallest epsilon")
        ax.set_xticks(rotation=45, ticks=x, labels=names, ha="right")
        ax.yaxis.grid(True)
        ax.set_title(f"Smallest epsilon for {self.hebb_name if not self.BP else ''} {self.name} on {self.dataset} for {self.attack} attack")
        ax.set_ylim(bottom=0)
        plt.savefig(f"{self.model.path}/attacks/{self.prefix}_classbar.png")
        plt.close()

        if self.no_hist:
            return
        mx = np.max(self.results["all"])
        bins = np.arange(0, mx, self.eps.min_delta * 1000)
        fig, ax = plt.subplots()
        H = ax.hist([self.results[c] for c in self.results.keys() if c != "all"], bins=bins, label=names[:-1], edgecolor="k", stacked=True)
        ax.set_xlabel(r"$\epsilon_{min}$")
        ax.set_ylabel("count")
        ax.set_title(f"Smallest epsilon for {self.hebb_name if not self.BP else ''} {self.name} on {self.dataset} for {self.attack} attack")
        ax.legend()
        ax.set_ylim(0, H[0].max())
        ax.set_xlim(0, self.fit.x.max())
        plt.savefig(f"{self.model.path}/attacks/{self.prefix}_hist.png")
        plt.close()

    def save_results(self):
        attack_info = DottableDefaultDict()
        if not self.BP: attack_info["hebb_name"] = self.model.hebb_name
        attack_info["name"] = self.model.name
        attack_info["seed"] = self.model.seed
        attack_info["test_acc"] = self.model.test_acc
        if not self.BP: attack_info["ratio"] = self.model.Hebbian.ratio
        attack_info["certainty"] = self.certainty
        attack_info["samples"] = self.samples
        attack_info["pad"] = self.pad
        attack_info["cutoff"] = self.cutoff
        attack_info["crit_eps"] = self.fit.crit_eps
        attack_info["MSE"] = self.fit.MSE
        attack_info["R2"] = self.fit.R2
        attack_info.update(self.fit.popt)

        output = f"\n{self.attack} attack results for {self.model.hebb_name if not self.BP else ''} {self.model.name} with certainty {self.certainty}%\n"
        output += "=" * len(output) + "\n"
        for key, val in attack_info.items():
            output += f"    {key:<25}: {val}\n"
        
        og = sys.stdout
        with open(f"{self.model.path}/attacks/{self.prefix}_results.txt", "w") as f:
            sys.stdout = f
            print(output)
        sys.stdout = og


def main(args=None):
    print()
    if args is None:
        args = attack_argparser()

    print(args)
    print(f"Device: {args.device}")

    if args.attack == "print":
        train_models.print_params(args)
    else:
        attacker = Attack_manager(args)
        attacker.do_attack(*args.name)

if __name__ == "__main__":
    main()