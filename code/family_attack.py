import utils.utils as utils
from utils.argparsing import attack_argparser
from model_attack import Attack_manager
from tqdm import tqdm
import os
import pandas as pd

famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

def record(manager):
    model = manager.model
    fit = manager.fit
    #* Set which parameters to record
    params = {}
    #* First record the hyperparameters of the model
    if not manager.BP:
        hebb = model.Hebbian
        params["hebb_name"]    = manager.hebb_name
        params["name"]         = manager.name
        params["K"]            = hebb.K
        params["width"]        = hebb.width
        params["p"]            = hebb.p
        params["k"]            = hebb.k
        params["delta"]        = hebb.delta
        params["hebb_eps0"]    = hebb.eps0
        params["hebb_lr_type"] = hebb.lr_type
        params["hebb_epochs"]  = hebb.num_epochs
        params["ratio"]        = hebb.ratio
        params["clas_eps0"]    = model.eps0
        params["clas_epochs"]  = model.num_epochs
        params["clas_lr_type"] = model.lr_type
    else:
        params["name"]    = manager.name
        params["K"]       = model.K
        params["width"]   = model.width
        params["eps0"]    = model.eps0
        params["lr_type"] = model.lr_type
    params["maxpool_kernel"] = model.maxpool_kernel
    params["power"]          = model.power
    params["activate"]       = model.activate
    params["test_acc"]       = model.test_acc

    #* record the attack parameters
    params["attack"]    = manager.attack
    params["certainty"] = manager.certainty
    params["samples"]   = manager.samples
    params["cutoff"]    = manager.cutoff
    params["pad"]       = manager.pad
    params["crit_eps"]  = fit.crit_eps
    params["MSE"]       = fit.MSE
    params["R2"]        = fit.R2
    params.update(fit.popt)
    
    file_name = f"{hebb_path(fam=manager.family)}/{manager.prefix}_complete_record.csv"

    df = pd.DataFrame([params])
    df.to_csv(file_name, mode="a", header=not os.path.isfile(file_name), index=False)

def attacked(model, hebb=None, args=None):
    df = pd.read_csv(f"{hebb_path(fam=args.family)}/{args.attack}_attack_record.csv")
    if hebb is None:
        return model in df["name"].values
    else:
        if hebb in df["hebb_name"].values:
            return model in df[df["hebb_name"] == hebb]["name"].values
    return False

def main(args=None):
    print()
    if args is None:
        args = attack_argparser()

    print(args)
    print("device:", args.device)

    args.family_attack = True

    assert hasattr(args, "family"), "Must specify a family"
    Manager = Attack_manager(args, show_pbar=False)

    if len(args.name) == 1:
        args.BP = True
        pbar = tqdm(existing_models(fam=args.family), desc=f"Attacking {args.family}")
        for model in pbar:
            # if not attacked(model, args=args):
            if Manager.do_attack(model):
                record(Manager)
                pbar.set_description(f"{model} Crit. eps: {Manager.fit.crit_eps:.5f}")

    else:
        args.BP = False
        pbar_hebb = tqdm(existing_models(fam=args.family), desc=f"Attacking {args.family}")
        for hebb in pbar_hebb:
            pbar_hebb.set_description(f"Attacking {args.family}: {hebb}")

            pbar = tqdm(existing_models(name=hebb, fam=args.family), desc=f"Attacking")
            for model in pbar:
                # if not attacked(model, hebb, args=args):
                if Manager.do_attack(hebb, model):
                    record(Manager)
                    pbar.set_description(f"{model} Crit. eps: {Manager.fit.crit_eps:.5f}")
            pbar.close()


if __name__ == "__main__":
    main()
