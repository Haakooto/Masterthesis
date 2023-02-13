"""
A file for analysing the results of the family of models by plotting.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from utils.utils import DottableDefaultDict as DDDict
from tempfile import TemporaryFile

def segment(df, names, keys, vals):
    """
    Takes a dataframe and makes many subframes based on the values of the keys.
    Arguments:
        df: the dataframe to segment
        names: the names of the subframes in the returned dictionary
        keys: the keys to segment on
        vals: the values to segment on
    Returns:
        A dictionary of dataframes
    """
    collection = DDDict()
    for i in range(len(keys)):
        collection[names[i]] = df[df[keys[i]] == vals[i]]
    collection.all = df
    return collection

# Read and add columns of MonkeyBrains
MBs = pd.read_csv("Models/MonkeyBrains/family_record.csv")
MBs.loc[MBs["lr_type"] != "exp", "eps_decay"] = 0
C = {"linear0.0": 0, "cosine0.0": 0.25, "exp4.0": 1, "exp2.5": 0.5, "exp3.25": 0.75}
MBs["lr_type_decay"] = [C[r[1]["lr_type"] + str(r[1]["eps_decay"])] for r in MBs.iterrows()]
MBs = MBs.sort_values("ratio")
MB_set = segment(MBs, ["weak", "ep075", "ep100", "ep125"], 
                      ["delta", "eps0", "eps0", "eps0"], 
                      [0.2, 75e-5, 100e-5, 125e-5])


def plot_hist_scatt(df, colorby="k"):
    print(df)
    
    fig, (h, s) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.90,
                        wspace=0.04,
                        hspace=0.04)

    H = h.hist(df["test_acc"], bins=np.linspace(0, 1, 21), orientation="horizontal", edgecolor="black")
    h.set_ylabel("Ratio of good neurons")
    h.set_xlabel("Number of models")
    h.set_ylim(0, 1)
    h.set_xlim(max(H[0]), 0)
    h.set_title(r"Histogram: bins$=0.05 \cdot i, i = [0, 1, \dots, 20] $")

    x = np.linspace(0, 1, len(df))

    scatter = s.scatter(x, df["test_acc"], c=df[colorby], cmap="cool")
    s.set_ylabel("Ratio of good neurons")
    s.set_xlabel("Relative rank")
    s.set_ylim(0, 1)
    s.yaxis.tick_right()
    s.yaxis.set_label_position("right")
    s.legend(*scatter.legend_elements(prop="colors"), title=f"value of {colorby}")
    s.set_title(f"Total number of models: {len(df)}")

    fig.suptitle(f"Ratio of good neurons in models")
    fig.savefig(f"figures/MonkeyBrains.png")

def id_subsets(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    ignore = ["seed", "good_model", "batch_size", "num_epochs", "delta", "width"]
    df = df.sort_values("ratio").drop(ignore, axis=1)
    changes = np.where(np.diff(df["ratio"]) > 0.025)[0]

    ch_mat = (np.repeat(changes[None, :], 2, axis=0) + 1) / len(df)
    ls = np.repeat([[0], [1]], len(changes), axis=1)
    ax.scatter(np.linspace(0, 1, len(df)), df["ratio"], color="blue")
    ax.plot(ch_mat, ls, "r--")

    ax.set_title("Big jumps in ratio of good neurons, eps0 = 75e-5, delta = 0.2")
    ax.set_xlabel("Relative rank")
    ax.set_ylabel("Ratio of good neurons")
    fig.savefig("rfigures/atio_scatter.png")

    for start, stop in zip(np.insert(changes, 0, 0), np.append(changes, len(df))):
        print(df.iloc[start:stop])
        print()

def covariance(df):
    df = df.drop(["seed", "good_model", "batch_size", "num_epochs", "width"], axis=1)
    cov = df.cov()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cov)
    ax.set_xticks(range(len(cov)))
    ax.set_yticks(range(len(cov)))
    ax.set_xticklabels(cov.columns)
    ax.set_yticklabels(cov.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.savefig("cfigures/ovariance.png")

def plot_hexbin(df, x, y):
    ax = df.plot.hexbin(x=x, y=y, C="ratio", gridsize=20, cmap="cool", reduce_C_function=np.mean)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Hexbin plot of {x} and {y}")
    fig = ax.get_figure()
    fig.savefig(f"figures/hexbin.png")

def fix_missing_columns(csv):
    with open(csv) as f, TemporaryFile("w+") as t:
        h, ln = next(f), len(next(f).split(","))
        header = h.strip().split(",")
        f.seek(0), next(f)
        header += range(ln)
        print(pd.read_csv(f, names=header))

def hist_scatt_classifiers(hebbian, colorby="power"):
    ogdf = pd.read_csv(f"Models/Generation3/{hebbian}/classifiers/family_record.csv").sort_values("test_acc")
    cutoff = 0.55
    df = ogdf[ogdf["test_acc"] >= cutoff]
    print(cutoff)
    print("0.001:", sum(df["eps0"] == 0.001) / sum(ogdf["eps0"] == 0.001))
    print("0.004:", sum(df["eps0"] == 0.004) / sum(ogdf["eps0"] == 0.004))
    print("0.008:", sum(df["eps0"] == 0.008) / sum(ogdf["eps0"] == 0.008))
    # print(df.groupby("lr_type").cumcount())

    return
    
    fig, (h, s) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.90,
                        wspace=0.04,
                        hspace=0.04)

    H = h.hist(df["test_acc"], bins=np.linspace(0, 1, 21), orientation="horizontal", edgecolor="black")
    h.set_ylabel("Test accuracy")
    h.set_xlabel("Number of models")
    h.set_ylim(0, 1)
    h.set_xlim(max(H[0]), 0)
    h.set_title(r"Histogram: bins$=0.05 \cdot i, i = [0, 1, \dots, 20] $")

    x = np.linspace(0, 1, len(df))

    scatter = s.scatter(x, df["test_acc"], c=df[colorby], cmap="cool")
    s.set_ylabel("Test accuracy")
    s.set_xlabel("Relative rank")
    s.set_ylim(0, 1)
    s.yaxis.tick_right()
    s.yaxis.set_label_position("right")
    s.legend(*scatter.legend_elements(prop="colors"), title=f"value of {colorby}")
    s.set_title(f"Total number of models: {len(df)}")

    fig.suptitle(f"Final test accuracy of {hebbian} classifiers")
    fig.savefig(f"figures/classifiers.png")

def generational_hist_scat(generation, colorby, cont=True):
    df = pd.read_csv(f"Models/{generation}/family_ranking.csv").sort_values("test_acc")
    df = df[df["test_acc"] >= 0.6]
    # df = df[df["clas_lr"] <= 0.002]
    # df = df[df["conv_rate"] >= 0.5]
    # df = df[df["MxP_kernel"] == 11]
    # df = df[df["width"] == 4]
    # df = df[df["power"] > 6]
    # df = df[df["p"] > 2]
    # df = df[df["conv_rate"] < 0.9]
    # df = df[df["clas_lr"] >= 0.001]

    if not cont: print(df)
    fig, (h, s) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.90,
                        wspace=0.04,
                        hspace=0.04)

    H = h.hist(df["test_acc"], bins=np.linspace(0, 1, 41), orientation="horizontal", edgecolor="black")
    h.set_ylabel("Test accuracy")
    h.set_xlabel("Number of models")
    h.set_ylim(0, 1)
    h.set_xlim(max(H[0]), 0)
    h.set_title(r"Histogram: bins$=0.05 \cdot i, i = [0, 1, \dots, 20] $")

    x = np.linspace(0, 1, len(df))

    cmap = "cool" if cont else "tab10"

    scatter = s.scatter(x, df["test_acc"], c=df[colorby], cmap=cmap)
    s.set_ylabel("Test accuracy")
    s.set_xlabel("Relative rank")
    s.set_ylim(0, 1)
    s.yaxis.tick_right()
    s.yaxis.set_label_position("right")
    s.legend(*scatter.legend_elements(prop="colors"), title=f"value of {colorby}")
    s.set_title(f"Total number of models: {len(df)}")

    fig.suptitle(f"Final test accuracy of all {generation} classifiers")
    fig.savefig(f"figures/{generation}_{colorby}.png")

if __name__ == "__main__":
    generational_hist_scat("ProjectMercury", "MxP_kernel")
    generational_hist_scat("ProjectMercury", "power")
    generational_hist_scat("ProjectMercury", "k")
    generational_hist_scat("ProjectMercury", "p")
    generational_hist_scat("ProjectMercury", "delta", False)
    generational_hist_scat("ProjectMercury", "K")
    generational_hist_scat("ProjectMercury", "clas_lr")
    generational_hist_scat("ProjectMercury", "hebb_lr")
    generational_hist_scat("ProjectMercury", "conv_rate")
    generational_hist_scat("ProjectMercury", "width")
    # fix_missing_columns("Models/Generation3/Bowie/classifiers/family_record.csv")
    # plot_hist_scatt(MB_set["all"], "p")
    # hist_scatt_classifiers("Carmen", "eps0")
    # plot_hist_scatt(MB_set["weak"])
    # plot_hist_scatt(weak, colorby="K")
    # plot_hexbin(weak, x="p", y="k")
    # plot_hist_scatt(weak)
    # id_subsets(weak)
    # covariance(MBs)
    # print(MB_set)