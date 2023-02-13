import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from tqdm import tqdm
import sys, os
os.chdir("./../")
print(os.getcwd())
from layers.classification import BioNet
from utils.utils import clas_path, existing_models

person = sys.argv[1]
img = torch.tensor(mpimg.imread(f"People/{person}.jpeg")) / 255
resize = T.Resize(32)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cpu")

family = "Activate"
models = []
for H in existing_models(fam=family):
    if H[5] != "4":
        continue
    try:
        for M in existing_models(fam=family, name=H):
            m = BioNet(path=clas_path(H, M, family))
            m.load_model(hebb_name=H, name=M, device=device)
            if m.num_epochs < 100:
                continue
            models.append(m)
    except:
        pass

print("Done loading models") 
probs = np.zeros((len(models), 10))
inp = resize(img.permute(2, 0, 1)).permute(1, 2, 0)[None, ...].to(device)
pbar = tqdm(models, desc="Predicting")
for i, model in enumerate(pbar):
    probs[i] = model(inp).exp().detach().cpu().numpy()[0]
    pbar.set_description(f"Model {i+1} says you're a {classes[np.argmax(probs[i])]:<10}")

font = {'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(2, 1, figsize=(8, 14))

mean = probs.mean(axis=0)
std = probs.std(axis=0)
mx = (std[np.argmax(mean)] + mean.max()) * 1.01

ax[0].imshow(inp[0])
ax[0].axis("off")
ax[0].set_title(f"The {len(models)} models agree,\n you're a{'n' if np.argmax(mean) < 2 else ''} {classes[np.argmax(mean)]}!", fontsize=30)
ax[1].set_title("Breakdown:")
ax[1].bar(np.arange(10), mean, yerr=std, capsize=5)
ax[1].set_xticks(rotation=45, ticks=np.arange(10), labels=classes, ha="right")
ax[1].set_ylim(0, mx)
ax[1].set_ylabel("Probability")

fig.savefig(f"People/pred_{person}.png")
plt.close()

