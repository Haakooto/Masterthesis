"""
File for training models, either hebbian layer or classifier layers
This file can be run directly, as with model_attack.py
"""

import torch
import os
import time
import pandas as pd

import utils.plot as plot
import utils.utils as utils
from utils.argparsing import train_argparser
from utils.dataloader import load_data, load_datas
from layers.local_learning import LocalLearning
from layers.classification import Classifier, BioNet, BPNet

from IPython import embed

#! See explaination in utils/utils.py for these
famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

def local_learning(args):
    """
    Local learning trainer. Will create new (directory for) hebbian layer and train it.
    If layer already exists, it can be trained further. The layer will be saved.
    After training, the weights are visualised and saved.
    """
    # Make family directory if not existing
    if not os.path.isdir(hebb_path()[:-1]):
        os.mkdir(hebb_path()[:-1])

    # Check if there already is a model with the same name saved in family folder
    model_existing = args.name in existing_models(fam=args.family)

    # initialize model with specified args
    torch.manual_seed(args.seed)
    Hebbian_layer = LocalLearning(path=hebb_path(args.name, fam=args.family), args=args)

    if model_existing:  # load model if it already exists
        if args.grid_search:
            print(f"Grid search: skipping {args.name}")
            #!###################################################################
            return  #! This is done to avoid prompting the user to load the model
            #!###################################################################

        if not args.force_retrain:
            if input(f"Hebbian layer with name {args.name} already exists. Load? [Y/n] ") != "n":
                Hebbian_layer.load_model()
                if args.train_more:
                    print("Setting layer state to untrained, will train more.")
                    Hebbian_layer.trained = False
                else:
                    print(f"Layer has already been trained. Skipping training.")
            else:
                print(f"All data saved about {args.name} will be overwritten if training is started.")
                os.system(f"rm {hebb_path(args.name, fam=args.family)}/saved_hebb_synapses/*")
        else:
            print(f"Forcefully overwriting all saved data about {args.name}.")
            os.system(f"rm {hebb_path(args.name, fam=args.family)}/saved_hebb_synapses/*")
    else:
        os.mkdir(hebb_path(args.name, fam=args.family))

    x = load_data(args, train=True).x

    # print(Hebbian_layer)
    Hebbian_layer.save_model()  # save model info before run
    Hebbian_layer.train_unsupervised(x, epochs=args.num_epochs)
    Hebbian_layer.draw_weights()
    plot.plot_ds(Hebbian_layer)
    if args.grid_search:
        family_record_local(Hebbian_layer, args.family)
        if args.save_all: return  # override deleting of bad models
        # delete model if it is not good.
        if not Hebbian_layer.good_model:
            if torch.rand(1).item() > 0.05:  #* save about 5% of bad models
                Hebbian_layer.purge(complete=True)

def family_record_local(model, fam):
    """
    Saves the paramaters and evaluation of each model to a dataframe shared by all models in the family.
    """
    params = ["name", "seed", "eps0", "lr_type", "K", "delta", "num_epochs", "p", "k", "width", "ratio", "good_model"]
    df = pd.DataFrame([{param: getattr(model, param) for param in params}])
    df.to_csv(f"{hebb_path(fam=fam)}/family_record.csv", mode="a", header=not os.path.isfile(f"{hebb_path(fam=fam)}/family_record.csv"), index=False)

def classify(args):
    """
    Trains the classifier layer of the model. The Hebbian layer must already be trained.
    """
    # Check if there already is a model with the same name saved in folder
    assert args.hebb_name in existing_models(fam=args.family), f"Hebbian model {args.hebb_name} does not exist in family {famy_path}."

    # if Hebbian layer has no classifiers, make a folder for them
    if not os.path.isdir(clas_path(args.hebb_name, fam=args.family)[:-1]):
        os.mkdir(clas_path(args.hebb_name, fam=args.family)[:-1])

    # Check if there already is a model with the same name saved in folder
    model_existing = args.name in existing_models(args.hebb_name, fam=args.family)

    torch.manual_seed(args.seed)
    Classifier_layer = BioNet(path=clas_path(args.hebb_name, args.name, fam=args.family), args=args)
    if model_existing:
        if not args.force_retrain:
            if input(f"Classifier layer for {args.hebb_name} with name {args.name} already exists. Load? [Y/n] ") != "n":
                Classifier_layer.load_model()
                if args.train_more:
                    print("Setting layer state to untrained, will train more.")
                    Classifier_layer.trained = False
                else:
                    print(f"Layer has already been trained. Skipping training.")
            else:
                print(f"All data saved about {args.name} will be overwritten if training is started.")
        else:
            print(f"Forcefully overwriting all saved data about {args.name}.")
    else:
        os.mkdir(clas_path(args.hebb_name, args.name, fam=args.family))

    train, test = load_datas(args)

    Classifier_layer.print()
    Teacher = Classifier(args, model=Classifier_layer)
    Teacher.teach(train, test, epochs=args.num_epochs)
    plot.plot_results(Classifier_layer)
    if args.grid_search:
        family_record_classify(Classifier_layer, args.family)

def family_record_classify(model, fam):
    """
    Saves the paramaters and evaluation of each model to a dataframe shared by all models in the family.
    """
    params = ["name", "seed", "eps0", "lr_type", "use_bias", "power", "num_epochs", "batch_size", "test_acc"]
    params += ["gamma",] if hasattr(model, "gamma") else []
    params += ["step_size",] if hasattr(model, "step_size") else []
    params += ["maxpool_kernel", "maxpool_stride", "activate"]

    df = pd.DataFrame([{param: getattr(model, param) for param in params}])
    df.to_csv(f"{clas_path(model.hebb_name, fam=fam)}/family_record.csv", mode="a", header=not os.path.isfile(f"{clas_path(model.hebb_name, fam=fam)}/family_record.csv"), index=False)

def endtoend(args):
    """
    Trains a classifier model end-to-end with backpropagation.
    """
    # Make family directory if not existing
    if not os.path.isdir(hebb_path(fam=args.family)[:-1]):
        os.mkdir(hebb_path(fam=args.family)[:-1])

    model_existing = args.name in existing_models(fam=args.family)

    torch.manual_seed(args.seed)
    Model = BPNet(path=hebb_path(args.name, fam=args.family), args=args)
    if model_existing:
        if not args.force_retrain:
            if input(f"BP model with name {args.name} already exists. Load? [Y/n] ") != "n":
                Model.load_model()
                if args.train_more:
                    print("Setting layer state to untrained, will train more.")
                    Model.trained = False
                else:
                    print(f"Layer has already been trained. Skipping training.")
            else:
                print(f"All data saved about {args.name} will be overwritten if training is started.")
        else:
            print(f"Forcefully overwriting all saved data about {args.name}.")
    else:
        os.mkdir(hebb_path(args.name, fam=args.family))

    train, test = load_datas(args)
    Model.print()

    Teacher = Classifier(args, model=Model)
    Teacher.teach(train, test, epochs=args.num_epochs)
    plot.plot_BP_results(Model)
    plot.E2E_hiddenW(Model)
    if args.grid_search:
        family_record_BP(Model, args.family)

def family_record_BP(model, fam):
    """
    Saves the paramaters and evaluation of each model to a dataframe shared by all models in the family.
    """
    params = ["name", "seed", "batch_size", "num_epochs", "K", "width", "stride", "activate", "power", "maxpool_kernel", "maxpool_stride", "use_bias", "eps0", "lr_type", "test_acc"]

    df = pd.DataFrame([{param: getattr(model, param) for param in params}])
    df.to_csv(f"{hebb_path(fam=fam)}/family_record.csv", mode="a", header=not os.path.isfile(f"{hebb_path(fam=fam)}/family_record.csv"), index=False)


def print_params(args):
    """
    Loads the model with the specified name and prints the parameters.
    Prints first the local layer, and if specified, the classification layer.
    """
    assert args.name[0] in existing_models(fam=args.family), f"Hebbian layer with name {args.name[0]} does not exist, cannot print parameters!"

    if len(args.name) == 2:
        assert args.name[1] in existing_models(args.name[0], fam=args.family), f"Classification layer with name {args.name[1]} does not exist, cannot print parameters!"

        Classifier_layer = BioNet(path=clas_path(args.name[0], args.name[1], fam=args.family))
        Classifier_layer.load_model(hebb_name=args.name[0], name=args.name[1])
        print(Classifier_layer)

    else:
        Hebbian_layer = LocalLearning(path=hebb_path(args.name[0], fam=args.family)).load_model(name=args.name[0], device=args.device)
        print(Hebbian_layer)
        Hebbian_layer.print()
        Hebbian_layer.save_model()

def rename_model(args):
    """
    Loads model and saves it with a new name, removing the one with the old name.
    Renaming is a bit tricky, but this function should do it.
    """
    assert args.name[0] in existing_models(fam=args.family), f"Hebbian layer with name {args.name[0]} does not exist, cannot load!"

    if len(args.name) == 2:
        assert args.name[1] in existing_models(args.name[0], fam=args.family), f"Classifier layer {args.name[1]} does not exist, cannot load!"

        new_name = input("New name: ")
        assert new_name not in existing_models(args.name[0], fam=args.family), "Layer with this name already exists, cannot rename!"

        os.rename(clas_path(*args.name, fam=args.family), clas_path(args.name[0], new_name, fam=args.family))

        model = BioNet()
        model.load_model(hebb_name=args.name[0], name=new_name)
        model.save_model()

    else:
        new_name = input("New name: ")
        assert new_name not in existing_models(fam=args.family), "Model with this name already exists, cannot rename!"

        os.rename(hebb_path(args.name[0], fam=args.family), hebb_path(new_name, fam=args.family))

        model = LocalLearning(path=hebb_path(new_name, fam=args.family))
        model.load_model(name=new_name)
        model.save_model()

        for name in existing_models(new_name, fam=args.family):
            model = BioNet(path=clas_path(new_name, name, fam=args.family))
            model.load_model(hebb_name=new_name, name=name)
            model.save_model()

def dummy(args):
    from tqdm import tqdm
    n = 25_000
    A = torch.randn((n, n)).to(args.device) / n
    x = torch.randn((n, 1)).to(args.device)
    code = input(f"\n\nGive code for {args.device}: ")
    for i in tqdm(range(5000), desc=f"Dummy on {args.device} with {code}"):
        x /= torch.linalg.norm(x)
        x = A @ x
    print(x[0])


def main(args=None):
    """
    Retrives parsed command line arguments, and calls the corresponding function.
    """
    if args is None:
        args = train_argparser()
    # if args.device is None: return None

    print(args)
    print(f"Device: {args.device}")

    if args.mode == "local-learning":
        local_learning(args)
    elif args.mode == "print":
        print_params(args)
    elif args.mode == "rename":
        rename_model(args)
    elif args.mode == "animate-weights":
        plot.animate_weights(LocalLearning(path=hebb_path(args.name, fam=args.family), args=args), args)
    elif args.mode == "classify":
        classify(args)
    elif args.mode == "endtoend":
        endtoend(args)
    elif args.mode == "dummy":
        dummy(args)
    else:
        raise ValueError(f"Invalid training mode: {args.mode}")

if __name__ == '__main__':
    main()
    # utils.Get_all_model_info()

