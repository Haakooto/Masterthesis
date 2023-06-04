import pandas as pd
import matplotlib as mpl
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from collections import defaultdict


def add_subplot_axes(ax, rect, axis=False):
    """
    useful utility for adding an axes to an axes. 
    Rect is list with [x, y, width, height] relative to parent axes 
    """
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
    """Thick black border around an axes."""

    def __init__(self, x_=0, y_=0):
        xmin = 0 + x_
        xmax = 1 - 2 * x_
        ymin = 0 + y_
        ymax = 1 - 2 * y_
        super().__init__((xmin, ymin), xmax, ymax, fill=False, ec="k")


class ControllButtonCollection(CheckButtons):
    """
    Connected check-buttons. Makes a list of checkbuttons
    The top button is parent, the following children
    If all children are checked, parent is fully checked
    If all children are unchecked, parent is unchecked
    If some children are checked, parent is partially checked
    Clicking partially checked parent will re-check all children
    Clicking fully checked parent will uncheck all children

    When a child button is clicked, first the value is flipped in the DataPlotter.sections['active']-dict is flipped
    Then the containers are called to show/hide the scatter-points affected

    There is one instance of this class for each parameter/section
    Every Container-instance is connected to every instance of this class
    """

    def __init__(self, name, tags, ax, flipper):
        super().__init__(ax, [name, ] +
                         sorted(list(tags)), [True, ] * (len(tags) + 1))
        self.name = name
        self.tags = tags
        self.ax = ax

        self._active = [True, ] * len(tags)

        # * Properties for parent
        self.rectangles[0].set_fill(True)
        self.rectangles[0].set_facecolor("black")
        self.rectangles[0].set_edgecolor("black")

        # * indent rectangle check-lines and text for children
        for kid in self.rectangles[1:]:
            kid.set_x(kid.get_x() + 0.2)
        for line_set in self.lines[1:]:
            for line in line_set:
                line.set_xdata(np.asarray(line.get_xdata()) + 0.2)
        for text in self.labels[1:]:
            text.set_x(text._x + 0.2)

        # * flips the boolean in 'sections['active']'-dict the corresponding to the value of the parameter (un)checked
        self.on_clicked(lambda lab: flipper(name, lab))

    def connect_controller(self, Controller, idx):
        """"""
        self.on_clicked(lambda lab: Controller.node_controller(
            self.name, idx, lab))

    def set_active(self, index):
        """
        The logic connecting each child to the parent
        Overwrites the method from CheckButtons
        called every time a button in the section is clicked
        further called the connected callbacks:
            flipper and Controller.node_controller
        """
        if index != 0:  # * children
            self._active[index - 1] = not self._active[index - 1]
            if sum(self._active) == 0:  # * all off, hide lines inside parent button
                for line in self.lines[0]:
                    line.set_visible(False)
            elif sum(self._active) == len(self._active):  # * all on, fill parent button
                self.rectangles[0].set_fill(True)
            else:
                # * all off, fill parent button partially
                self.rectangles[0].set_fill(False)
                for line in self.lines[0]:
                    line.set_visible(True)
            super().set_active(index)

        else:  # * main button
            # * If all children active, disable all children
            if sum(self._active) == len(self._active):
                for i, line_set in enumerate(self.lines):
                    if i == 0:  # * parent button has no connected callbacks itself, but virtually clicks all the other children
                        for line in line_set:
                            line.set_visible(False)
                            self.rectangles[i].set_fill(False)
                    else:
                        if self._active[i-1]:
                            super().set_active(i)
                            self._active[i-1] = False
            else:  # * Else enable all that are disabled
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


class DataPlotter:
    Containers = {}
    """
    Object for containing the axes, and sections. Binds the Container-objects to the ControllButtonCollection-objects
    """

    def __init__(self, params):
        # * init figure and axes
        self.fig, self.main_ax = plt.subplots(1, 1, figsize=(10, 10))
        self.main_ax.axis("off")
        plt.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.07)

        # * two main axes, one for scatter and one for hyperparameter selection
        self.pick_ax = add_subplot_axes(self.main_ax, [0.8, 0, 0.2, 1])
        self.pick_ax.add_artist(AxBorder(0.02, 0.01))
        self.data_ax = add_subplot_axes(self.main_ax, [0, 0, 0.8, 1], True)
        self.data_ax.set_ylabel("Test accuracy")
        self.data_ax.set_xlabel("Critical epsilon (robustness)")

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)

        # * Make the button sections
        """
        Will evenly space the button sections vertically
        Each section has one button for each value of the parameter
        The value of each parameter is displayed next to the button
            This is done by 'tags'
            Also used to decide how many buttons are needed in ControllButtonCollection
        Everything is stored in 'sections' a dict with 2 entries per param:
            'buttons' is the ControllButtonCollection-object
            'active' is a dict with a boolean value for each value of the parameter
        """
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
        self.legend_texts = {}  # * enable pickable legend for hiding entire Containers
        for i, text in enumerate(self.legend.get_texts()):
            text.set_picker(10)
            self.legend_texts[text] = i
        plt.show()

    def flip(self, property, value):
        self.sections[property]["active"][value] = not self.sections[property]["active"][value]

    def on_pick(self, event):
        # * More legend picking
        if event.artist in self.legend_texts:
            index = self.legend_texts[event.artist]
            dots = DataPlotter.Containers[index].dots
            vis = dots.get_visible()
            dots.set_visible(not vis)
            event.artist.set_alpha(1.0 if not vis else 0.2)


class Node:
    def __init__(self, kwargs):
        kwargs = {str(k): str(v) for k, v in kwargs.items()}
        self.__dict__.update(kwargs)

    def get(self, prop):
        return self.__dict__[prop]


class Container:
    idx = 0

    def __init__(self, collection, Plotter, colorbar="None"):
        self.idx = Container.idx
        Container.idx += 1
        Plotter.Containers[self.idx] = self

        # * 'parse' the collection-dict
        self.total_df = collection["df"]
        self.hyperparams = collection["hyperparams"]
        self.legname = collection["legname"]
        self.all_params = collection["hyperparams"] + \
            collection["other_params"]
        self.df = collection["df"][self.all_params]
        self.Plotter = Plotter
        self.bp = collection["bp"]
        self.fam = collection["fam"]
        self.prefix = collection["prefix"]

        [Plotter.sections[param]["buttons"].connect_controller(  # * Connect each of the ControllButtonCollection-objects to the Container
            self, i) for i, param in enumerate(self.hyperparams)]  # * (each CBC must be premade for each parameter)

        # * each for in df gets made into a Node, stored in this dict. Each node is a model
        self.nodes = {i: Node(dict(row)) for i, row in self.df.iterrows()}
        # * bool-mat with col for every node, and col for every hyperparam
        self.active = np.ones(
            (len(self.hyperparams), len(self.nodes)), dtype=bool)

        # * Plot the data and intialize the hover annotation
        if colorbar == "ratio":
            self.dots = Plotter.data_ax.scatter(
                self.df["crit_eps"], self.df["test_acc"], c=self.df["ratio"], label=self.label())
            plt.colorbar(self.dots)
        elif colorbar == "samples":
            self.dots = Plotter.data_ax.scatter(
                self.df["crit_eps"], self.df["test_acc"], c=self.df["samples"], label=self.label())
            plt.colorbar(self.dots)
        else:
            self.dots = Plotter.data_ax.scatter(
                self.df["crit_eps"], self.df["test_acc"], color=collection["color"], label=self.label())
        self.hover_ind = None
        self.annot = Plotter.data_ax.annotate("", xy=(0, 0), xytext=collection["hover_pos"], textcoords="offset points",
                                              bbox=dict(
                                                  boxstyle="round", fc="w", ec=collection["color"]),
                                              arrowprops=dict(arrowstyle="->"))
        # * Connect the event callbacks
        Plotter.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        Plotter.fig.canvas.mpl_connect("button_press_event", self.click)

    def label(self):
        # * Show visible model-count in legend
        return f"{self.legname}: {sum(self.active.all(axis=0))} models"

    def node_controller(self, property, pidx, value):
        # * A node is only visible is all of its hyperparams are active
        if property in self.hyperparams:  # * BP-models do not have p and k, and not affected by them
            new_state = self.Plotter.sections[property]["active"][value]
            for i, node in self.nodes.items():
                # * find the nodes which have the value of the parameter in question
                if node.get(property) == value:
                    self.active[pidx, i] = new_state

            # * partially (un)hide nodes with not all properties active
            self.dots.set_alpha(np.where(self.active.all(axis=0), 1, 0.05))
            self.Plotter.legend.texts[self.idx].set_text(
                self.label())  # * update legend with new count

    def update_annot(self, ind):
        """
        Update and show the annotation when hovering. Put the desired info into it, and place it close to the dot
        """
        # ! The ind is a dict, and can contain multiple points. Always pick the first in list
        idxs = set(ind["ind"])
        hiddens = set(np.where(~self.active.all(axis=0))[0])
        shown = idxs.difference(hiddens)
        if len(shown) == 0:
            return False
        idx = list(shown)[0]
        self.hover_ind = idx
        self.annot.xy = self.dots.get_offsets()[idx]
        if self.bp:
            text = f'{self.fam}\n{self.nodes[idx].get("name"):<20}\n{self.nodes[idx].get("samples"):<4} samples'
        else:
            text = f'{self.fam}\n{self.nodes[idx].get("hebb_name"):<10}\n{self.nodes[idx].get("name"):<10}\n{self.nodes[idx].get("samples"):<4} samples'
        self.annot.set_text(text)
        return True

    def hover(self, event):
        if event.inaxes == self.Plotter.data_ax:
            cont, ind = self.dots.contains(event)
            if cont:
                if self.update_annot(ind):
                    self.annot.set_visible(True)
            else:
                self.annot.set_visible(False)
                self.hover_ind = None

        self.Plotter.fig.canvas.draw_idle()

    def click(self, event):
        if event.inaxes == self.Plotter.data_ax:
            if self.hover_ind is not None:
                print("You can not download images from my server!")
                print("They would have shown the convolution layer and robustness curve of the model you clicked")
                return
            
                hebb_fig = plt.figure(
                    num=self.Plotter.fig.number + self.idx * 2 + 1)
                hebb_fig.clf()
                attack_fig = plt.figure(
                    num=self.Plotter.fig.number + self.idx * 2 + 2)
                attack_fig.clf()

                hebb_ax = hebb_fig.add_subplot(1, 1, 1)
                attack_ax = attack_fig.add_subplot(1, 1, 1)

                node = self.nodes[self.hover_ind]
                name = node.get("name")

                if self.bp:
                    path = f"/Models/{self.fam}/{name}/"
                    syn_path = path + "visualised_weights.png"
                    path += f"attacks/{self.prefix}/curve.png"
                    tmp_syn = "./syns_bp.png"
                    tmp_att = "./attack_bp.png"

                else:
                    hebb_name = node.get("hebb_name")
                    path = f"/Models/{self.fam}/{hebb_name}/"
                    syn_path = path + "visualised_hebbian_synapses.png"
                    path += "classifiers/" + name + \
                        f"/attacks/{self.prefix}/curve.png"
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


class Collector:
    def __init__(self):
        self.data = []

    def build_dataset(self, fam: str, prefix: str, legname: str, color: str, BP: bool, pgd: bool, pos: tuple, bad_fit_kill: float=0.1):
        file_name = f"{fam}_{prefix}"
        df = pd.read_csv(f"./csvs/{file_name}_complete_record.csv")
        df = df.drop(df[df["R2"] < bad_fit_kill].index).reset_index(drop=True)
        hyper = ["K", "width", "power", "activate", "maxpool_kernel"]
        other = ["name", "test_acc", "samples",
                 "crit_eps", "MSE", "R2", "attack", "certainty"]
        if not BP:
            hyper.append("p")
            hyper.append("k")
            other.append("hebb_name")
            other.append("ratio")
        if pgd:
            other.append("proj_size")
            other.append("step_size")
        collector = {"df": df,
                     "fam": fam,
                     "bp": BP,
                     "hyperparams": hyper,
                     "other_params": other,
                     "legname": legname,
                     "color": color,
                     "hover_pos": pos,
                     "prefix": prefix,
                     }
        self.data.append(collector)
        return collector

    @property
    def tags(self):
        tags = defaultdict(set)
        for D in self.data:
            for param in D["hyperparams"]:
                for p in D["df"][param].unique():
                    tags[param].add(p)
        return tags

    def contain(self, DP, cbar="None"):
        for dataset in self.data:
            Container(dataset, DP, colorbar=cbar)


def main():

    DataCollector = Collector()

    DataCollector.build_dataset("Activate",    "fgsm_C08"           , "LL: fgsm_C8", color="blue"  , BP=False, pgd=False, pos=(20, -50))
    # DataCollector.build_dataset("Activate",    "fgsm_C06"           , "LL: fgsm_C6", color="yellow", BP=False, pgd=False, pos=(20, -50))
    # DataCollector.build_dataset("Activate",    "fgsm_S1000"         , "LL: fgsm_S" , color="red"   , BP=False, pgd=False, pos=(20, -10))
    DataCollector.build_dataset("ActivateBP",  "fgsm_C08"           , "BP: fgsm_C8", color="green" , BP=True , pgd=False, pos=(20, 20))
    # DataCollector.build_dataset("ActivateBP",  "fgsm_C06"           , "BP: fgsm_C6", color="orange", BP=True , pgd=False, pos=(20, 70) )
    # DataCollector.build_dataset("ActivateBP",  "fgsm_S1000"         , "BP: fgsm_S" , color="black" , BP=True , pgd=False, pos=(20, 70) )

    # DataCollector.build_dataset("Activate",    "pgd_C08_s012_e0001" , "LL: pgd"    , color="green" , BP=False, pgd=True , pos=(20, -50))
    # DataCollector.build_dataset("Activate",    "pgd_C08_s014_e00005", "LL: pgd"    , color="blue"  , BP=False, pgd=True , pos=(20, 10))
    # DataCollector.build_dataset("Activate",    "pgd_C07_s01_e00005" , "LL: pgd"    , color="red"   , BP=False, pgd=True , pos=(20, -100))
    # DataCollector.build_dataset("ActivateBP",  "pgd_C08_s012_e0001" , "BP: pgd"    , color="green" , BP=True , pgd=True , pos=(20, -50))
    # DataCollector.build_dataset("ActivateBP",  "pgd_C08_s014_e00005", "BP: pgd"    , color="red"   , BP=True , pgd=True , pos=(20, 10))
    # DataCollector.build_dataset("ActivateBP",  "pgd_C07_s01_e00005" , "BP: pgd"    , color="blue"  , BP=True , pgd=True , pos=(20, -100))

    # DataCollector.build_dataset("Ensamblers",  "fgsm_C08"           , "LL: fgsm"   , color="blue"  , BP=False, pgd=False, pos=(20, -50))
    # DataCollector.build_dataset("Ensamblers",  "pgd_C08_s012_e0001" , "LL: pgd"    , color="green" , BP=False, pgd=True , pos=(20, -80))
    # DataCollector.build_dataset("EnsamblersBP","fgsm_C08"           , "BP: fgsm"   , color="red"   , BP=True , pgd=False, pos=(20, 00))
    # DataCollector.build_dataset("EnsamblersBP","pgd_C08_s012_e0001" , "BP: pgd"    , color="orange", BP=True , pgd=True , pos=(20, 50))

    DP = DataPlotter(DataCollector.tags)
    DataCollector.contain(DP)
    DP.show()


if __name__ == "__main__":
    main()
