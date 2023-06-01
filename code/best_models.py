import torch
import pandas as pd
import matplotlib.pyplot as plt
import utils.utils as utils
from layers.classification import BPNet
from tqdm import tqdm

family_path, hebb_path, clas_path, existing_models = utils.path_utils_pack

LL = pd.read_csv("Models/Activate/fgsm_08_complete_record.csv")
BP = pd.read_csv("Models/ActivateBP/fgsm_08_complete_record.csv")
# LL = pd.read_csv("myLLmodels.csv")

# LL = LL.loc[(LL["K"] >= 15)]
# LL = LL.loc[(LL["R2"] >= 0.99) & (LL["samples"] > 1000)]
# BP = BP.loc[(BP["R2"] >= 0.99) & (BP["samples"] > 1000)]

# plt.scatter(LL["test_acc"], LL["crit_eps"], marker="*", alpha=0.8, label=f"LL: {len(LL)} models")
# plt.scatter(BP["test_acc"], BP["crit_eps"], marker="v", alpha=0.7, label=f"BP: {len(BP)} models")
# plt.legend()

# # s = plt.scatter(LL["test_acc"], LL["crit_eps"], marker="*", alpha=0.8, label=f"LL: {len(LL)} models", c=LL["hebb_name"], cmap="viridis")
# # plt.legend(*s.legend_elements(prop="colors"), title="hebbian")

# plt.xlabel("Test accuracy")
# plt.ylabel("Critical epsilon")
# # plt.ylim(0, max(max(LL["crit_eps"]), max(BP["crit_eps"])) * 1.05)
# plt.title("Robustness vs accuracy for LL and BP models")
# # plt.show()
# plt.savefig("Figures/acc_vs_crit_newnew.png")

# print("Very robust LL models:")
# print(LL.loc[(LL["crit_eps"] > 0.03)])
print(LL.loc[LL["test_acc"].idxmax()])
# print(BP.loc[BP["test_acc"].idxmax()])
# print("Very robust BP models:")
# print(BP.loc[(BP["crit_eps"] > 0.025)])

# print(BP.loc[(BP["crit_eps"] > 0.03) & (BP["test_acc"] > 0.55)])
