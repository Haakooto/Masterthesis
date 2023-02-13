#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import torchvision.transforms as T
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tqdm import tqdm
import os, sys
sys.path.append('./../')

from layers.classification import BioNet
from utils.utils import clas_path, existing_models

def predict(path, name=None):
    img = torch.tensor(imread(path)) / 255  # preprocess image
    resize = T.Resize(32)
    print(f"Recieved image{'' if name is None else ' ' + name}")

    curr_path = os.getcwd()
    os.chdir("./../")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = torch.device("cuda")
    family = "Activate"
    # Load models
    models = []
    H_models = existing_models(fam=family)
    np.random.shuffle(H_models)
    for H in H_models:
        try:
            for M in existing_models(fam=family, name=H):
                m = BioNet(path=clas_path(H, M, family))
                m.load_model(hebb_name=H, name=M, device=device)
                if m.num_epochs < 100:
                    continue
                models.append(m)
        except:
            pass
        if len(models) > 1000: break
    
    if len(models) == 0: return False
    
    # prepare containers and input
    probs = np.zeros((len(models), 10))
    inp = resize(img.permute(2, 0, 1)).permute(1, 2, 0)[None, ...].to(device)
    pbar = tqdm(models, desc="Predicting")
    for i, model in enumerate(pbar):
        probs[i] = model(inp).exp().detach().cpu().numpy()[0]

    os.chdir(curr_path)
    font = {'weight' : 'bold',
            'size'   : 20}
    mpl.rc('font', **font)

    fig, ax = plt.subplots(2, 1, figsize=(9, 14))

    mean = probs.mean(axis=0)
    std = probs.std(axis=0)
    mx = (std[np.argmax(mean)] + mean.max()) * 1.01

    ax[0].imshow(inp[0].cpu().numpy())
    ax[0].axis("off")
    ax[0].set_title(f"The {len(models)} models agree,\n {'this' if name is None else name} is a{'n' if np.argmax(mean) < 2 else ''} {classes[np.argmax(mean)]}!", fontsize=30)
    ax[1].set_title("Breakdown:")
    ax[1].bar(np.arange(10), mean, yerr=std, capsize=5)
    ax[1].set_xticks(rotation=45, ticks=np.arange(10), labels=classes, ha="right")
    ax[1].set_ylim(0, mx)
    ax[1].set_ylabel("Probability")

    pred = f"Predicts/pred{'' if name is None else '_'  + name}.png"
    fig.savefig(pred)
    plt.close()

    return pred  # return path to the prediction image

if __name__ == '__main__':
    path = sys.argv[1]
    name = None if len(sys.argv) < 3 else sys.argv[2]
    print(predict(path, name))

