import pandas as pd
import matplotlib as mpl
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
import numpy as np
import os


def add_subplot_axes(ax, rect, axis=False):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    if not axis:
        subax.axis("off")
    return subax


class AxBorder(mpl.patches.Rectangle):
    def __init__(self, x_=0, y_=0):
        xmin = 0 + x_
        xmax = 1 - 2 * x_
        ymin = 0 + y_
        ymax = 1 - 2 * y_
        super().__init__((xmin, ymin), xmax, ymax, fill=False, ec="k")


class ControllButtonCollection(CheckButtons):
    def __init__(self, name, tags, ax, flipper):
        super().__init__(ax, [name, ] +
                         sorted(list(tags)), [True, ] * (len(tags) + 1))
        self.name = name
        self.tags = tags
        self.ax = ax

        self._active = [True, ] * len(tags)

        self.rectangles[0].set_fill(True)
        self.rectangles[0].set_facecolor("black")
        self.rectangles[0].set_edgecolor("black")

        for kid in self.rectangles[1:]:
            kid.set_x(kid.get_x() + 0.2)
        for line_set in self.lines[1:]:
            for line in line_set:
                line.set_xdata(np.asarray(line.get_xdata()) + 0.2)
        for text in self.labels[1:]:
            text.set_x(text._x + 0.2)

        self.on_clicked(lambda lab: flipper(name, lab))

    def connect_controller(self, Controller, idx):
        self.on_clicked(lambda lab: Controller.node_controller(
            self.name, idx, lab))

    def set_active(self, index):
        if index != 0:
            self._active[index - 1] = not self._active[index - 1]
            if sum(self._active) == 0:
                for line in self.lines[0]:
                    line.set_visible(False)
            elif sum(self._active) == len(self._active):
                self.rectangles[0].set_fill(True)
            else:
                self.rectangles[0].set_fill(False)
                for line in self.lines[0]:
                    line.set_visible(True)
            super().set_active(index)

        else:
            for i, line_set in enumerate(self.lines):
                if i == 0:
                    for line in line_set:
                        line.set_visible(True)
                        self.rectangles[i].set_fill(True)
                else:
                    if not self._active[i-1]:
                        super().set_active(i)
                        self._active[i-1] = True

        self.ax.figure.canvas.draw()


class Node:
    def __init__(self, kwargs):
        kwargs = {str(k): str(v) for k, v in kwargs.items()}
        self.__dict__.update(kwargs)

    def get(self, prop):
        return self.__dict__[prop]


class DataPlotter:
    def __init__(self, params):
        # * init figure and axes
        self.fig, self.main_ax = plt.subplots(1, 1, figsize=(10, 10))
        self.main_ax.axis("off")
        plt.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.07)

        # * two main axes, one for scatter and one for hyperparameter selection
        self.data_ax = add_subplot_axes(self.main_ax, [0, 0, 0.8, 1], True)
        self.pick_ax = add_subplot_axes(self.main_ax, [0.8, 0, 0.2, 1])
        self.pick_ax.add_artist(AxBorder(0.02, 0.01))
        self.data_ax.set_xlabel("Test accuracy")
        self.data_ax.set_ylabel("Critical epsilon (robustness)")

        # * Make the button sections
        bottom = 0
        count = sum([len(params[param]) + 1 for param in params])
        self.sections = {}
        for param in params:
            tags = params[param]
            height = (len(tags) + 1) / count
            ax = add_subplot_axes(self.pick_ax, [0.05, bottom, 0.95, height])
            self.sections[param] = {"buttons": ControllButtonCollection(param, tags, ax, self.flip),
                                    "active": {str(tag): True for tag in tags}}
            bottom += height

    def show(self):
        self.legend = self.data_ax.legend()
        plt.show()

    def flip(self, property, value):
        self.sections[property]["active"][value] = not self.sections[property]["active"][value]

class Container:
    idx = 0
    def __init__(self, collection, Plotter, bp=False):
        self.idx = Container.idx
        Container.idx += 1

        self.total_df    = collection["df"]
        self.hyperparams = collection["hyperparams"]
        self.legname     = collection["legname"]
        self.all_params  = collection["hyperparams"] + collection["other_params"]
        self.df          = collection["df"][self.all_params]
        self.Plotter     = Plotter
        self.bp          = bp

        [Plotter.sections[param]["buttons"].connect_controller(
            self, i) for i, param in enumerate(self.hyperparams)]

        self.nodes = {i: Node(dict(row)) for i, row in self.df.iterrows()}
        self.active = np.ones((len(self.hyperparams), len(self.nodes)), dtype=bool)
        # * Plot the data and intialize the hover annotation
        self.dots = Plotter.data_ax.scatter(self.df["test_acc"], self.df["crit_eps"], color="red" if bp else "blue", label=self.label())
        self.hover_ind = None
        self.annot = Plotter.data_ax.annotate("", xy=(0, 0), xytext=(20, 20 if bp else -40), textcoords="offset points",
                                           bbox=dict(boxstyle="round", fc="w"),
                                           arrowprops=dict(arrowstyle="->"))
        # * Connect the event callbacks
        Plotter.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        Plotter.fig.canvas.mpl_connect("button_press_event", self.click)
    
    def label(self):
        return f"{self.legname}: {sum(self.active.all(axis=0))} models"

    def node_controller(self, property, pidx, value):
        if property in self.hyperparams:
            new_state = self.Plotter.sections[property]["active"][value]
            for i, node in self.nodes.items():
                if node.get(property) == value:
                    self.active[pidx, i] = new_state

            self.dots.set_alpha(np.where(self.active.all(axis=0), 1, 0.1))
            self.Plotter.legend.texts[self.idx].set_text(self.label())

    def update_annot(self, ind):
        # ! The ind is a dict, and can contain multiple points. Always pick the first in list
        idx = ind["ind"][0]
        self.hover_ind = idx
        self.annot.xy = self.dots.get_offsets()[idx]
        if self.bp:
            text = f'{self.nodes[idx].get("name"):<20}'
        else:
            text = f'{self.nodes[idx].get("hebb_name"):<10}\n{self.nodes[idx].get("name"):<10}'
        self.annot.set_text(text)

    def hover(self, event):
        if event.inaxes == self.Plotter.data_ax:
            cont, ind = self.dots.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
            else:
                self.annot.set_visible(False)
                self.hover_ind = None

        self.Plotter.fig.canvas.draw_idle()

    def click(self, event):
        if event.inaxes == self.Plotter.data_ax:
            if self.hover_ind is not None:
                hebb_fig = plt.figure(num=self.Plotter.fig.number + self.idx * 2 + 1)
                hebb_fig.clf()
                attack_fig = plt.figure(num=self.Plotter.fig.number + self.idx * 2 + 2)
                attack_fig.clf()

                hebb_ax = hebb_fig.add_subplot(1, 1, 1)
                attack_ax = attack_fig.add_subplot(1, 1, 1)

                node = self.nodes[self.hover_ind]
                name = node.get("name")

                if self.bp:
                    path = f"/Models/ActivateBP/{name}/"
                    syn_path = path + "visualised_weights.png"
                    path += "attacks/fgsm_08_eps_curve.png"
                    tmp_syn = "./syns_bp.png"
                    tmp_att = "./attack_bp.png"

                else:
                    hebb_name = node.get("hebb_name")
                    path = f"/Models/Activate/{hebb_name}/"
                    syn_path = path + "visualised_hebbian_synapses.png"
                    path += "classifiers/" + name + "/attacks/fgsm_08_eps_curve.png"
                    tmp_syn = "./syns.png"
                    tmp_att = "./attack.png"
                
                neuro_download(syn_path, tmp_syn)
                neuro_download(path, tmp_att)

                hebb_ax.imshow(plt.imread(tmp_syn))
                attack_ax.imshow(plt.imread(tmp_att))
                hebb_ax.axis("off")
                attack_ax.axis("off")

                plt.show()

def neuro_download(path_from, path_to):
    cmd = f"scp neuroai:~/local_learning_robustness/code{path_from} {path_to}"
    os.system(cmd)


def main():
    LL = pd.read_csv("myLLmodels.csv")
    BP = pd.read_csv("myBPmodels.csv")

    LL_collector = {"df": LL,
                    "hyperparams": ["K", "width", "p", "k", "power", "activate", "maxpool_kernel"],
                    "other_params":  ["hebb_name", "name", "ratio", "test_acc", "samples", "crit_eps", "MSE", "R2"],
                    "legname": "LL"}
    BP_collector = {"df": BP,
                    "hyperparams": ["K", "width", "power", "activate", "maxpool_kernel"],
                    "other_params":  ["name", "test_acc", "samples", "crit_eps", "MSE", "R2"],
                    "legname": "BP"}

    button_tags = {param: LL[param].unique()
                   for param in LL_collector["hyperparams"]}

    DP = DataPlotter(button_tags)
    Container(LL_collector, DP)
    Container(BP_collector, DP, True)

    DP.show()


if __name__ == "__main__":
    main()
