from glob import glob
import os
import pandas as pd
from train_models import existing_models
from layers.classification import BPNet
import pickle

from IPython import embed

def remove_double_header(filename):
    """ Sometimes the header is duplicated. This removes the second header. """
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        f.write(lines[0])
        for line in lines[1:]:
            if line == lines[0]:
                print("Double header found in", filename)
                continue
            f.write(line)

def add_maxpool_state(filename):
    """ Adds the maxpool state to the csv file.
        Models starting with C has no maxpool, i.e. kernel_size=1, stride=1
        Models starting with P has maxpool, i.e. kernel_size=11, stride=2
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        lines[0] = lines[0].strip()
        lines[0] += ",mp_kernel,mp_stride\n"
        f.write(lines[0])

        for line in lines[1:]:
            line = line.strip()
            if line[0] == "C":
                line += ",1,1\n"
            elif line[0] == "P":
                line += ",11,2\n"
            f.write(line)

def glob_models(fam):
    """ Returns a list of all csv files in the generation folder, recuvrsively. """
    models = existing_models(fam=fam)
    if models[-1] == "__pycache__":
        models = models[:-1]
    csvs = [file
            for path, subdir, files in os.walk(f"Models/{fam}")
            for file in glob(os.path.join(path, "*.csv")) 
            if ("fgsm" not in file and "pgd" not in file)]
    return models, csvs  #* Skip family_ranking.csv

def change_csvs(fam):
    """ Makes changes to the csv files. """
    _, files = glob_models(fam)
    for file in files[1:]:
        print("removing double header in", file)
        remove_double_header(file)
        # add_maxpool_state(file)

def gather_family_record(fam="ProjectMercury"):
    """
    All the classifier models are stored seperately, under each of the hebbian models.
    Thin function will gather all the variable hyperparameters for all the classifier models,
    so that they can be compared directly.
    The dataframe is saved as 'family_ranking.csv' in the generation folder.
    """
    living_models, csvs = glob_models(fam)
    hebbians, clas = pd.read_csv(csvs[0]), csvs[1:]
    classers = {file.split("/")[2]: pd.read_csv(file) for file in clas}

    hebb_cols_conv = {"name"   : "hebb_name",
                      "eps0"   : "hebb_lr",
                      "lr_type": "hebb_lr_type",
                      "K"      : "K",
                      "delta"  : "delta",
                      "p"      : "p",
                      "k"      : "k",
                      "width"  : "width",
                      "ratio"  : "ratio",
                      }
    clas_cols_conv = {"name"     : "clas_name",
                      "eps0"     : "clas_lr",
                      "lr_type"  : "clas_lr_type",
                      "power"    : "power",
                      "maxpool_kernel": "MxP_kernel",
                      "maxpool_stride": "MxP_stride",
                      "test_acc" : "test_acc",
                      "activate" : "activate",
                      }
    collect = []
    for hebbian_name in living_models:
        if hebbian_name not in classers.keys():
            continue
        hebbian = hebbians[hebbians["name"] == hebbian_name][hebb_cols_conv.keys()].rename(columns=hebb_cols_conv).reset_index(drop=True)
        for classer in classers[hebbian_name].iterrows():
            clas = classer[1][clas_cols_conv.keys()].rename(clas_cols_conv).to_frame().T.reset_index(drop=True)
            collect.append(hebbian.combine_first(clas))
    df = pd.concat(collect, axis=0, ignore_index=True, join="outer")
    column_order = [10, 6, 8, 9, 4, 5, 16, 0, 11, 12, 7, 13, 3, 1, 2, 14, 15]
    df = df[df.columns[column_order]]
    fname = f"Models/{fam}/family_ranking.csv"
    df.to_csv(fname, index=False)
    print("Saved", fname)

def gather_BP_models():
    """I forgot to record family records for the BP models.
    This function will collect that info."""
    models = glob("Models/ProjectMercuryBP/*")
    collect = []
    params = ["name", "seed", "batch_size", "num_epochs", "K", "width", "stride", "power", "maxpool_kernel", "maxpool_stride", "use_bias", "eps0", "lr_type"]
    for model in models:
        Net = BPNet(path=model)
        Net.load_model(name=model.split("/")[-1])
        dump = {param: Net.__dict__[param] for param in params}
        with open(f"{Net.path}/training_results.pkl", "rb") as f:
            results = pickle.load(f)
            last = {key: value[-1] for key, value in results.items()}
        collect.append({**dump, **last})
    df = pd.DataFrame(collect)
    df.to_csv("Models/ProjectMercuryBP/family_ranking.csv", index=False)

def find_untrained_classifiers(fam, epochs=100, count=85):
    """ Finds all the untrained classifiers in the family. """
    _, csvs = glob_models(fam)
    hebbians, clas = pd.read_csv(csvs[0]), csvs[1:]
    classers = {file.split("/")[2]: pd.read_csv(file) for file in clas}
    hebber_count = 0
    for hebbian_name in classers.keys():
        hebber_count += 1
        cnt = 0
        for classer in classers[hebbian_name].iterrows():
            cnt += 1
            if classer[1]["num_epochs"] < epochs:
                print(f"{hebbian_name} {classer[1]['name']} has not been trained for {epochs} epochs, only {classer[1]['num_epochs']}")
        if cnt != count:
            print(f"{hebbian_name} does not have {count} classifiers, only {cnt}")
    print(hebber_count)

def ratio_control(fam):
    """ Checks that all the hebbian models have a ratio of 0.9 or higher. """
    _, csvs = glob_models(fam)
    hebbians, clas = pd.read_csv(csvs[0]), csvs[1:]
    classers = {file.split("/")[2]: pd.read_csv(file) for file in clas}
    hebb_dict = hebbians.to_dict("index")
    hebb_dict = {hebb_dict[key]["name"]: hebb_dict[key] for key in hebb_dict.keys()}
    cnt = 0
    for hebbian_name in classers.keys():
        cnt += 1
        assert hebb_dict[hebbian_name]["ratio"] >= 0.8
    print(cnt)

def tmp_main():
    df = pd.read_csv("Models/Activate/family_record.csv")
    df = df.loc[df["ratio"] >= 0.8]

    print(df[df["name"] == "K20_w4_32"])


if __name__ == "__main__":
    # change_csvs("Activate")
    # find_untrained_classifiers("Activate")
    # ratio_control("Activate")
    # df = pd.read_csv("Models/ProjectMercury/family_ranking.csv").sort_values(by="test_acc", ascending=False)
    # print(df)
    # gather_BP_models()
    # df = pd.read_csv("Models/ProjectMercuryBP/family_ranking.csv").sort_values(by="test_accuracy", ascending=False)
    # print(df)
    # tmp_main()

    gather_family_record("Activate")
    # fam = "Activate"
    # living_models, csvs = glob_models(fam)
    # hebbians, clas = pd.read_csv(csvs[0]), csvs[1:]
    # for cla in clas:
    #     remove_double_header(cla)
    # classers = {file.split("/")[2]: pd.read_csv(file) for file in clas}

    # embed()