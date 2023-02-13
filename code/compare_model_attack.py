"""
File that will attack multiple models in a family, and visualize the results in a single plot.
"""

import utils.utils as utils
from utils.argparsing import attack_argparser
from model_attack import Attack_manager
from tqdm import tqdm
import matplotlib.pyplot as plt

famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

# LL = ["LL", "Activate", "K10_w4_46", "MX07_elu08"]
# BP = ["BP", "ActivateBP", "BP_K20_w04_MX11_rexp"]

LL = {"leg": "LL", "family": "Activate", "hebb_name": "K20_w4_34", "name": "MX09_relu12", "BP": False, "Ac": "firebrick", "Fc": "darkblue"}
BP = {"leg": "BP", "family": "ActivateBP", "name": "BP_K20_w04_MX11_rexp", "BP": True, "Ac": "green", "Fc": "darkblue"}

def main(models, args=None):
    print()
    if args is None:
        args = attack_argparser()

    print(args)
    print("device:", args.device)

    Manager = Attack_manager(args)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for model in models:
        leg = model["leg"]
        Manager.family = model["family"]
        Manager.BP = model["BP"]
        if Manager.BP:
            Manager.name = model["name"]
        else:
            Manager.hebb_name = model["hebb_name"]
            Manager.name = model["name"]

        assert Manager.do_attack(plot=True)

        ax.plot(Manager.fit.x, Manager.fit.y, "o-", color=model["Ac"], label=f"{leg}-model. Certainty threshold: {args.certainty}")
        ax.plot(Manager.fit.x, Manager.fit.eval, "--", color=model["Fc"], label=f"Fit. MSE: {Manager.fit.MSE:.5f}, R2: {Manager.fit.R2:.5f}")
        ax.axvline(Manager.fit.crit_eps, color=model["Ac"], linestyle="--", label=f"Critical epsilon: {Manager.fit.crit_eps:.5f}")

    ax.set_title(f"Comparison of robustness")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.savefig(f"Figures/compare_crit_eps_new.png")
    plt.close()


if __name__ == "__main__":
    main([LL, BP])
