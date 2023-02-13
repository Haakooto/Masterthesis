import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import glob
import pickle
import utils.telebot as telebot
import utils.fitting as fitting
import utils.utils as utils


def format_nice_plot(axis=False):
    """
    Very tighly packs the layout, sets font properties, and turns off axis
    """
    plt.subplots_adjust(left=0.05,
                        bottom=0.05,
                        right=0.95,
                        top=0.90,
                        wspace=0.04,
                        hspace=0.04)
    font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}
    if not axis: plt.axis("off")
    plt.rc("font", **font)


def plot_patches(patches):
    """Plots the patches for a single image, as well as the image itself"""
    true = patches.pop(0)
    # first_patch = patches[5]
    # print(first_patch.reshape(8, 8, 3).transpose((1, 0, 2)))
    plt.imshow(np.asarray(true))
    plt.savefig("the_true.png")
    plt.clf()

    ppd = int(np.sqrt(len(patches)))
    fig, axs = plt.subplots(nrows=ppd, ncols=ppd)
    for i, ax in enumerate(axs.flat):
        ax.imshow(np.asarray(patches[i].reshape((8,8,3)).transpose(1, 0, 2)))
        ax.axis("off")
    plt.savefig("patches.png")
    plt.clf()

def plot_patche_classf(true, patches):
    """Was used for testting early implementation of patch classification. Now Hebbian model does this"""
    plt.imshow(np.asarray(true))
    plt.savefig("the_true_classf.png")
    plt.clf()

    ppd = int(np.sqrt(patches.size(0)))
    fig, axs = plt.subplots(nrows=ppd, ncols=ppd)
    # __import__("IPython").embed()
    for i, ax in enumerate(axs.flat):
        # patch = patches[i].detach().cpu().reshape((8,8,3))
        ax.imshow(np.asarray(patches[i].reshape((8,8,3)).transpose(1, 0)))
        ax.axis("off")
    plt.savefig("patches_classf.png")
    plt.clf()

def plot_ds(model):
    """Plots the ds of the hebbian model during training. ds is analogous to loss, and goes to 0 for good models"""
    fig, ax = plt.subplots()
    x = np.linspace(0, model.num_epochs, len(model._ds_evolution))
    ax.plot(x, np.log10(model._ds_evolution))
    ax.set_title(f"DS during training of {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"$\log_{10}(max|ds|)$")
    plt.savefig(f"{model.path}/ds_evolve.png")
    plt.clf()


def animate_weights(model, args):
    """Animates the weights of the model during training"""
    data_files = sorted(glob.glob(f"{model.path}/saved_hebb_synapses/*.pkl"), key=lambda x: int(x.split("_")[-1].split(".")[0]))

    fig, ax = plt.subplots()
    # ax.set_title(f"Hebbian synapse activations of {args.name}")
    # ax.axis("off")

    def animate(i):
        ax.clear()
        epoch = data_files[i].split("/")[-1].split(".")[0].split("_")[-1]
        with open(data_files[i], "rb") as f:
            synapeses = pickle.load(f)
        model.hebb_synapses = synapeses
        HM = model.draw_weights(anim=True)
        nc = np.max(np.absolute(HM))
        im = ax.imshow(HM, cmap="bwr", vmin=-nc, vmax=nc)
        ax.set_title(f"Hebbian synapse activations of {args.name} at epoch {int(epoch)}.\n{model.unconverged_neurons() * 100:.2f}% of synapses have converged.")

        # ax.set_title(f"Hebbian synapse activations of {args.name} at epoch {int(epoch)}")
        ax.axis("off")

    model.load_model()
    ani = animation.FuncAnimation(fig, animate, interval=1000, frames=len(data_files))
    name = f"{model.path}/hebbian_evolution.gif"
    ani.save(name, writer="pillow")

def plot_results(model):
    """Plots the resulting accuracies and losses"""
    with open(f"{model.path}/training_results.pkl", "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots()
    ax.plot(np.log10(results["train_loss"]), label="Train loss")
    ax.plot(np.log10(results["test_loss"]), label="Test loss")
    ax.legend()
    ax.set_title(f"Loss during training of {model.hebb_name} {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("log(Loss)")
    plt.savefig(f"{model.path}/loss.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(results["train_accuracy"], label="Train accuracy")
    ax.plot(results["test_accuracy"], label="Test accuracy")
    ax.legend()
    ax.set_title(f"Accuracy during training of {model.hebb_name} {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"{model.path}/accuracy.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(results["train_certainty"], label="Train certainty")
    ax.plot(results["test_certainty"], label="Test certainty")
    ax.legend()
    ax.set_title(f"certainty during training of {model.hebb_name} {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("certainty")
    plt.savefig(f"{model.path}/certainty.png")
    plt.clf()

def plot_BP_results(model):
    """Plots the resulting accuracies and losses"""
    with open(f"{model.path}/training_results.pkl", "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots()
    ax.plot(np.log10(results["train_loss"]), label="Train loss")
    ax.plot(np.log10(results["test_loss"]), label="Test loss")
    ax.legend()
    ax.set_title(f"Loss during training of {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("log(Loss)")
    plt.savefig(f"{model.path}/loss.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(results["train_accuracy"], label="Train accuracy")
    ax.plot(results["test_accuracy"], label="Test accuracy")
    ax.legend()
    ax.set_title(f"Accuracy during training of {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"{model.path}/accuracy.png")
    plt.clf()


    fig, ax = plt.subplots()
    ax.plot(results["train_certainty"], label="Train certainty")
    ax.plot(results["test_certainty"], label="Test certainty")
    ax.legend()
    ax.set_title(f"certainty during training of {model.name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("certainty")
    plt.savefig(f"{model.path}/certainty.png")
    plt.clf()

def E2E_hiddenW(model):
        assert not model._needs_parameter_loading, "Model needs to be given parameters before drawing weights"
        conv = model.Conv2d

        neurons = int(np.sqrt(conv.out_channels))

        shape = (conv.in_channels, conv.kernel_size[0], conv.kernel_size[1])
        rearrange = lambda synapse: synapse.permute((1, 2, 0))

        HM = np.zeros(
            (shape[1] * neurons, shape[2] * neurons, conv.in_channels))
            
        normed_W = (conv.weight if shape[0]==1 else utils.minmaxnorm(conv.weight)).detach().cpu()
        
        cnt = 0
        fig, ax = plt.subplots()
        for y in range(neurons):
            for x in range(neurons):
                activity = rearrange(normed_W[cnt])
                HM[y * shape[2]:(y+1) * shape[2],
                   x * shape[1]:(x+1) * shape[1], :] = activity
                cnt += 1

        nc = np.max(np.absolute(HM))  #* normalize colorbar
        im = ax.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
        ax.set_title(f"Weights of {model.name}.")
        format_nice_plot()

        name = f"{model.path}/visualised_weights.png"
        plt.savefig(fname=name)
        plt.clf()


def ranged_attack_plot_single(results, args, path):
    """
    Plots the results of a ranged attack for a single model
    """
    fig, ax = plt.subplots()
    eps = np.asarray(list(results.keys())).reshape(-1, 1)
    acc = np.asarray(list(results.values()))

    fit = fitting.fit_genlog(results)

    ax.plot(eps, acc, "bo-", label="Measured accuracy")
    ax.plot(eps, fit.eval, "r--", label=f"Logistic fit. MSE: {fit.MSE:.5f}, R^2: {fit.R2:.5f}")
    ax.axvline(x=fit.crit_eps, color="k", linestyle="--", label=f"Critical epsilon: {fit.crit_eps:.5f}")

    ax.set_title(f"Accuracy of {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.savefig(f"{path}/attacks/{args.eps.mode}_{args.attack}.png")
    plt.clf()

def smallest_few_attack_plot(results, args, path):
    fig, axs = plt.subplots(nrows=args.examples, ncols=2, figsize=(12, 5*args.examples))
    if args.examples == 1:
        axs = np.asarray([axs])
    for ex, data in results.items():
        if args.BP:
            d1 = data["image"].transpose((1, 2, 0))
            d2 = data["adversarial"].transpose((1, 2, 0))
        else:
            d1 = data["image"]
            d2 = data["adversarial"]
        axs[ex, 0].imshow(d1)
        axs[ex, 0].axis("off")
        axs[ex, 0].set_title(f"Original: class {data['target']}")

        axs[ex, 1].imshow(d2)
        axs[ex, 1].axis("off")
        axs[ex, 1].set_title(f"Adversarial: prediction {data['pred']}")
        axs[ex, 1].annotate(f"Epsilon: {round(data['eps'], 5)}.  Certainty: {round(data['certainty'], 3)}",
                            (0, 0), (0, -10),
                            xycoords='axes fraction',
                            textcoords='offset points',
                            va='top')
    plt.subplots_adjust(left=0.02,
                        bottom=0.04,
                        right=0.98,
                        top=0.90,
                        wspace=0.02,
                        hspace=0.15)
    fig.suptitle(f"Adversarial examples for {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack", fontsize=16)
    plt.savefig(f"{path}/attacks/smallest_{args.attack}.png")
    plt.clf()

def class_boundary_plot(results, args, path):
    fig, ax = plt.subplots()

    sorter = np.argsort(list(results.keys()))

    x = np.arange(len(results))
    means = np.asarray([np.mean(result) for result in results.values()])[sorter]
    stds = np.asarray([np.std(result) for result in results.values()])[sorter]
    classes = np.asarray([name for name in results.keys()])[sorter]
    names = np.asarray([f"Class {i}" for i in classes])
    names[0] = "All"

    ax.bar(x, means, yerr=stds, align="center", alpha=0.5, ecolor="black", capsize=10)
    for i, n in enumerate(classes):
        ax.text(i, 0, f"{len(results[n])}", color="blue", fontweight="normal", ha="center", va="bottom")

    ax.set_ylabel("Smallest epsilon")
    ax.set_xticks(rotation=45, ticks=x, labels=names, ha="right")
    ax.yaxis.grid(True)
    ax.set_title(f"Smallest epsilon for {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack")
    ax.set_ylim(bottom=0)
    plt.savefig(f"{path}/attacks/class_{args.attack}.png")
    plt.clf()

    # mx = np.max([np.max(results[c]) for c in classes[1:] if len(results[c]) != 0]) / 3
    mx = 0.1
    bins = np.arange(0, mx, args.eps.min_delta * 100)
    fig, ax = plt.subplots()
    H = ax.hist([results[c] for c in classes[1:]], bins=bins, label=names[1:], edgecolor="k", stacked=True)
    ax.set_xlabel(r"$\epsilon_{min}$")
    ax.set_title(f"Smallest epsilon for {args.hebb_name if not args.BP else ''} {args.name} on {args.dataset} for {args.attack} attack")
    ax.legend()
    ax.set_ylim(0, H[0].max())
    plt.savefig(f"{path}/attacks/class_{args.attack}_hist.png")
    plt.clf()