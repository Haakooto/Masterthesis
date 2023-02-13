"""
File with the following classes:
    BioNet:     Classification layer of Hebbian models
    BPNet:      Backpropagation model
    Classifier: Used to train the two models above
"""

import torch
from torch import nn, optim
from torch.nn import functional as FF
from einops import rearrange
import numpy as np
import pickle, sys, os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from layers.local_learning import LocalLearning

class BioNet(nn.Module):
    def __init__(self, path, args=None):
        super(BioNet, self).__init__()
        self.path = path
        self.chw = False  #* The model takes images in the format (H, W, C) instead of (C, H, W)

        if args is not None:
            self.reset_model(args)
            self._needs_parameter_loading = False
        else:
            self._needs_parameter_loading = True

    def reset_model(self, args=None):
        if args is not None:
            self.__dict__.update(args.__dict__)

            assert self.dataset in ("CIFAR", "MNIST"), "Dataset not supported."
            self.num_classes = 10
            self.num_epochs = 0

            self.eps0 = self.scheduler.eps0
            self.lr_type = self.scheduler.lr_type
            self.__dict__.update(self.scheduler.kwargs)

            self.Hebbian = LocalLearning("/".join(self.path.split("/")[:4]))
            self.Hebbian = self.Hebbian.load_model(self.hebb_name, device=self.device)

            self.init_activation_function()

            self.maxPool = nn.MaxPool2d(self.maxpool_kernel, stride=self.maxpool_stride)
            self.poolshape_in = (self.Hebbian.num_hidden, self.Hebbian._patches_pr_dim, self.Hebbian._patches_pr_dim)
            self.poolshape_out = ((self.Hebbian._patches_pr_dim - self.maxPool.kernel_size) // self.maxPool.stride + 1) ** 2
            self.classifier = nn.Linear(self.Hebbian.num_hidden * self.poolshape_out, self.num_classes, bias=self.use_bias, device=self.device)

            self.time_of_training = []
            self.trained = False
            self.aborted = False

        else:
            self.Hebbian.reset_model()
            for layer in self.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def maxPoolShape(self, x):
        """Legacy function"""
        assert False, "This function should not be used anymore."
        return self.maxPool(x.reshape(x.shape[0], *self.poolshape_in))

    def save_model(self):
        """
        Saves the classifier model to disk. All parameters are in the __dict__.
        All params except the Hebbian model are saved.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before saving."

        self.print()

        tmp = self.Hebbian
        self.Hebbian = f"The Hebbian model {self.hebb_name} is saved and loaded seperately."
        tmp_act = self.activation
        self.activation = "some activation function"

        self.classifier.cpu()
        
        with open(f"{self.path}/classifier_parameters.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)

        self.Hebbian = tmp
        self.activation = tmp_act
        self.classifier.to(self.device)

    def load_model(self, hebb_name=None, name=None, device=None):
        """
        Loads the model parameters
        Loads both the BP-layer, and the Hebbian layer.
        Most of the dealing with names is for option of renaming the model.
        """
        if self._needs_parameter_loading:
            assert name is not None, "Empty model needs to be given a name to load parameters from."
            self.name = name
            assert hebb_name is not None, "Empty model needs to be given a Hebbian model to load parameters from."
            self.hebb_name = hebb_name

        tmp_path = self.path

        with open(f"{self.path}/classifier_parameters.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        if name is not None:
            self.name = name
        if hebb_name is not None:
            self.hebb_name = hebb_name
            sections = self.path.split("/")
            sections[3] = self.hebb_name
            self.path = "/".join(sections)
        if device is not None:
            self.device = device
        self.path = tmp_path

        if not hasattr(self, "activate"):
            self.activate = "relu"

        self.init_activation_function()

        self.Hebbian = LocalLearning("/".join(self.path.split("/")[:4]))
        self.Hebbian = self.Hebbian.load_model(name=self.hebb_name, verbose=False, device=self.device)  #* I don't trust my horrible code
        self.classifier.to(self.device)

    def draw_weights(self, path=""):
        if not os.path.isdir(f"{self.path}/weights{path}"):
            os.mkdir(f"{self.path}/weights{path}")

        weights = self.classifier.weight.detach().cpu().numpy()

        for clas in range(self.num_classes):
            fig, ax = plt.subplots()
            HM = weights[clas].reshape(self.Hebbian.num_hidden, self.poolshape_out)
            nc = np.max(np.absolute(HM))  #* normalize colorbar
            im = ax.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
            ax.set_title(f"Weights of {self.name} for class {clas}")
            ax.set_xlabel("Hidden units")
            ax.set_ylabel("Pooled units")

            cbar = ax.figure.colorbar(im, ax=ax)

            name = f"{self.path}/weights{path}/class{clas}.png"
            plt.savefig(fname=name)
            plt.clf()

    def __str__(self):
        print(self.Hebbian)

        out = f"\nParameters of classifier layer of {self.name}:\n"
        out += "=" * len(out) + "\n"
        for key, val in self.__dict__.items():
            if key[0] == "_": continue
            if key == "Hebbian":
                val = f"hebbian layer {self.hebb_name}"
            elif key == "classifier":
                val = f"classifier matrix of shape {val.weight.shape}"
            out += f"    {key:<25}: {val}\n"
        return out

    def print(self):
        # self.Hebbian.print()
        og = sys.stdout
        with open(f"{self.path}/readable_parameters.txt", "w") as f:
            sys.stdout = f
            print(self)
        sys.stdout = og

    def init_activation_function(self):
        """Activation function. Added later, so some back-compatibility is needed."""
        if self.activate == "exp":
            self.activation = torch.exp
        elif self.activate == "rexp":
            self.activation = lambda x: torch.where(x > 0, torch.exp(x) - 1, 0)
        else:
            f = getattr(FF, self.activate)
            self.activation = lambda x: f(x) ** self.power

    def forward(self, x):                              #* (imgs, img_dim, img_dim,  channels)
        x = self.Hebbian(x)                            #* (imgs, hidden,  patches)
        x = x.reshape(x.shape[0], *self.poolshape_in)  #* (imgs, hidden,  patches_pr_dim, patches_pr_dim)
        x = self.maxPool(x)                            #* (imgs, hidden,  poolshape_out)
        # x = FF.gelu(x) ** self.power                   #* (imgs, hidden,  poolshape_out)
        x = self.activation(x)                         #* (imgs, hidden,  poolshape_out)
        x = torch.flatten(x, 1)                        #* (imgs, hidden * poolshape_out)
        x = self.classifier(x)                         #* (imgs, classes)
        return FF.log_softmax(x, dim=1)

    def hidden_layer(self, x):  #* (imgs, img_dim, img_dim,  channels)
        """Returns the hidden layer output"""
        return self.Hebbian(x)  #* (imgs, hidden, patches)

class BPNet(nn.Module):
    def __init__(self, path, args=None):
        super().__init__()
        self.path = path
        self.chw = True  #* The model takes images in the format (C, H, W) instead of (H, W, C)

        if args is None:
            self._needs_parameter_loading = True
        else:
            self.reset_model(args)

    def reset_model(self, args=None):
        if args is not None:
            self._needs_parameter_loading = False
            self.__dict__.update(args.__dict__)

            assert self.dataset in ("CIFAR", "MNIST"), "Dataset not supported."
            self.channels = 3 if self.dataset == "CIFAR" else 1
            self.img_dim = 32 if self.dataset == "CIFAR" else 28
            self.num_classes = 10
            self.num_epochs = 0

            self.eps0 = self.scheduler.eps0
            self.lr_type = self.scheduler.lr_type
            self.__dict__.update(self.scheduler.kwargs)

            self.num_hidden = self.K ** 2

            self.init_activation_function()

            self.Conv2d = nn.Conv2d(self.channels, self.num_hidden, self.width, stride=self.stride, bias=self.use_bias, device=self.device)
            self.MaxPool = nn.MaxPool2d(self.maxpool_kernel, stride=self.maxpool_stride)
            self.class_shape = ((self.img_dim - self.width) // self.stride + 1 - self.maxpool_kernel) // self.maxpool_stride + 1
            self.classifier = nn.Linear((self.K * self.class_shape) ** 2, self.num_classes, bias=self.use_bias, device=self.device)

            self.time_of_training = []
            self.trained = False
            self.aborted = False

        else:
            for layer in self.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def forward(self, x):                       #* (imgs, img_dim, img_dim, channels)
        x = rearrange(x, "n h w c -> n c h w")  #* (imgs, channels, img_dim, img_dim)
        x = self.Conv2d(x)                      #* (imgs, hidden, patches_pr_dim, patches_pr_dim)
        x = self.MaxPool(x)                     #* (imgs, hidden, poolshape_out, poolshape_out)
        # x = FF.relu(x) ** self.power
        x = self.activation(x)
        x = torch.flatten(x, 1)                 #* (imgs, hidden * poolshape_out ** 2)
        x = self.classifier(x)                  #* (imgs, classes)
        output = FF.log_softmax(x, dim=1)
        return output

    def init_activation_function(self):
        """Activation function. Added later, so some back-compatibility is needed."""
        if self.activate == "exp":
            self.activation = torch.exp
        elif self.activate == "rexp":
            self.activation = lambda x: torch.where(x > 0, torch.exp(x) - 1, 0)
        else:
            f = getattr(FF, self.activate)
            self.activation = lambda x: f(x) ** self.power

    def hidden_layer(self, x):                         #* (imgs, img_dim, img_dim, channels)
        """Returns the hidden layer output"""
        x = rearrange(x, "n h w c -> n c h w")         #* (imgs, channels, img_dim, img_dim)
        x = self.Conv2d(x)                             #* (imgs, hidden, patches_pr_dim, patches_pr_dim)
        x = rearrange(x, "n H Px Py -> n H (Px Py)")  #* (imgs, hidden, patches)
        return x

    def save_model(self):
        assert self.trained, "Model needs to be trained before saving."
        self.print()

        tmp = self.activation
        self.activation = "some activation function"
        self.Conv2d.cpu()
        self.classifier.cpu()

        with open(f"{self.path}/BP_parameters.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)

        self.activation = tmp
        self.Conv2d.to(self.device)
        self.classifier.to(self.device)

    def load_model(self, name=None, device=None):
        """
        Loads the model parameters
        Loads both the BP-layer, and the Hebbian layer.
        Most of the dealing with names is for option of renaming the model.
        """
        if self._needs_parameter_loading:
            assert name is not None, "Empty model needs to be given a name to load parameters from."
            self.name = name

        with open(f"{self.path}/BP_parameters.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        if name is not None:
            self.name = name
        if device is not None:
            self.device = device
        
        self.Conv2d.to(self.device)
        self.classifier.to(self.device)

        if not hasattr(self, "activate"):
            self.activate = "relu"
        
        self.init_activation_function()

    def __str__(self):
        out = f"\nParameters of BP-model {self.name}:\n"
        out += "=" * len(out) + "\n"
        for key, val in self.__dict__.items():
            out += f"    {key:<25}: {val}\n"
        return out

    def print(self):
        og = sys.stdout
        with open(f"{self.path}/readable_BP_parameters.txt", "w") as f:
            sys.stdout = f
            print(self)
        sys.stdout = og

class Classifier:
    def __init__(self, args, model):

        self.__dict__.update(args.__dict__)
        self.model = model

        self.results = defaultdict(list)

    def teach(self, train_set, test_set=None, epochs=0):
        self.no_test = test_set is None
        model = self.model
        assert not model._needs_parameter_loading, "Model needs to be given parameters before training."

        if model.trained:
            print(f"Model {model.name} has already been trained. Skipping training.")
            return

        model.time_of_training.append(datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))

        self.optimizer = optim.AdamW(model.parameters(), lr=self.scheduler.eps0)
        self.scheduler = self.scheduler(self.optimizer, epochs)  #* Gets a torch.optim.lr_scheduler object from utils.py

        new_epochs = model.num_epochs + epochs
        pbar = tqdm(range(1, epochs+1))

        train_set.set_chw(model.chw)  #* Makes sure the dataset is in the correct format
        train_set.to(model.device)  #* Makes sure the dataset is on the correct device

        print(f"\nRunning supervised training")
        print(f"Epochs: {epochs}, images: {len(train_set)}, batch size: {self.batch_size}")
        if not self.no_test:
            test_set.set_chw(model.chw)
            test_set.to(model.device)
            print(f"Testing on {len(test_set)} images")
        print()

        try:
            for epoch in pbar:
                self.train(train_set)
                self.test(None if self.no_test else test_set)
                self.scheduler.step()

                model.num_epochs += 1
                pbar.set_description(f"BP. Loss: {self.results['train_loss'][-1]:.5f}. Accuracies: {self.results['train_accuracy'][-1]:.3f}, {self.results['test_accuracy'][-1]:.3f}")
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt. Aborting training early, after {epoch}/{epochs} epochs. Total epochs: {model.num_epochs}")
            model.aborted = True
        else:
            model.aborted = False
        pbar.set_description(f"BP. Loss: {self.results['train_loss'][-1]:.5f}. Accuracies: {self.results['train_accuracy'][-1]:.3f}, {self.results['test_accuracy'][-1]:.3f}")

        model.test_acc = self.results["test_accuracy"][-1]
        model.trained = True
        model.save_model()
        self.save_results()

    def train(self, data):
        self.model.train()
        train_loss = 0
        correct = 0
        certainty = 0

        data.shuffle()

        for bidx in range(0, len(data), self.batch_size):
            x = data.x[bidx: bidx + self.batch_size]
            target = data.targets[bidx: bidx + self.batch_size]

            self.optimizer.zero_grad()
            output = self.model(x)

            loss = FF.nll_loss(output, target, reduction='sum')
            loss.backward()
            self.optimizer.step()

            c, pred = output.max(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            train_loss += loss.item()
            certainty += c.exp().sum().item()

        self.results["train_loss"].append(train_loss / len(data))
        self.results["train_accuracy"].append(correct / len(data))
        self.results["train_certainty"].append(certainty / len(data))

    def test(self, data):
        if data is None: return 0

        self.model.eval()
        test_loss = 0
        correct = 0
        certainty = 0

        data.shuffle()

        with torch.no_grad():
            for bidx in range(0, len(data), self.batch_size):
                x = data.x[bidx: bidx + self.batch_size]
                target = data.targets[bidx: bidx + self.batch_size]

                output = self.model(x)
                test_loss += FF.nll_loss(output, target, reduction='sum').item()
                c, pred = output.max(dim=1, keepdim=False)
                correct += pred.eq(target).sum().item()
                certainty += c.exp().sum().item()

        self.results["test_loss"].append(test_loss / len(data))
        self.results["test_accuracy"].append(correct / len(data))
        self.results["test_certainty"].append(certainty / len(data))

    def save_results(self):
        with open(self.model.path + "/training_results.pkl", "wb") as f:
            pickle.dump(self.results, f)

