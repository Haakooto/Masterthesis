import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from utils.plot import format_nice_plot
import sys
from datetime import datetime
import utils.telebot as telebot
import pickle
import os
import shutil
import einops

class LocalLearning:
    """
    unsupervised training of the first layer for generating representations in images
    """
    def __init__(self, path, args=None):
        """
        Initializes the model. If args is None, the model needs to be loaded by load_model().
        The path argument is a bodge and is getting quite annoying to deal with.
        A hypothetical future refactoring should fix this.
        """
        self.path = path

        if args is not None:
            self.reset_model(args)
            self._needs_parameter_loading = False
        else:
            self._needs_parameter_loading = True

    def reset_model(self, args=None):
        """
        Initializes the model. This is called by __init__ if args is not None
        All arguments in args are copied directly to the model.
        """
        if args is not None:
            self.__dict__.update(args.__dict__)   #* copy all arguments to self
            self.make_private()                   #* rename some attributes to not show when printing

            self.num_epochs = 0                   #* number of epochs trained for

            #* Get lr params for printing. Not used again
            self.lr_type = self.scheduler.type()  #* A torch.optim.lr_scheduler object set in utils/utils.py
            self.eps0 = self.scheduler.eps0

            self.num_hidden = self.K ** 2         #* number of hidden units in Ky x Kx array
            self.patch_pixels = self.width ** 2 * self.channels

            #* pre-make some tensors to avoid re-allocating memory in the training loop
            self._g_i = torch.zeros(self.num_hidden, self.batch_size)
            self._indexer = torch.arange(self.batch_size)

            #* The weights of the model
            self.hebb_synapses = torch.Tensor(self.num_hidden,   #* initialize synapses
                                            self.patch_pixels, #* size KxK x patch_pixels
                                            ).normal_(         #* normal-distributed
                                                        self.mu,
                                                        self.sigma,
                                                        )
            self.time_of_training = []
            self._ds_evolution = []
            self.trained = False
            self.aborted = False

        else:
            self.hebb_synapses.normal_(self.mu, self.sigma)

    def make_private(self):
        """ Rename some attributes to not show when printing """
        attrs = ["force_retrain", "mode", "prec", "save_every", "save_all", "grid_search"]

        for attr in attrs:
            self.__dict__[f"_{attr}"] = self.__dict__.pop(attr)

    def draw_weights(self, anim=False, neurons_x=np.nan, neurons_y=np.nan):
        """
        Draws the weights of the synapses. If neurons_x and neurons_y are not given,
        all synapses will be shown. Otherwise, only neurons_x synapses are shown
        in the x-direction, and neurons_y synapses shown in the y-direction.

        neurons_x and neurons_y are clipped to the range [1, self.K].
        non-positive inputs are invalid, and clips to self.K

        Have to rearrange the weights because of the way the patches are rearranged in the time-optimized training loop.

        If anim is True, the weights are drawn in a matplotlib animation by returning the matrix before drawing.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before drawing weights"

        #* clip inputs to valid range, default to self.K
        neurons_x = int(neurons_x) if (
            neurons_x := np.nanmin([neurons_x, self.K])) > 0 else self.K
        neurons_y = int(neurons_y) if (
            neurons_y := np.nanmin([neurons_y, self.K])) > 0 else self.K
        
        cnt = 0
        shape = (self.width, self.width, self.channels)
        rearrange = lambda synapse: synapse.reshape(*shape).transpose(0, 1)

        HM = np.zeros(
            (self.width * neurons_y, self.width * neurons_x, self.channels))

        #* When plotting weights for MNIST-models, the synapses are not touched, but for CIFAR they must be minmaxnormed
        synapses = self.hebb_synapses.t()
        normed_synapses = (synapses if self.channels==1 else self.minmaxnorm(synapses)).detach().cpu()

        fig, ax = plt.subplots()
        for y in range(neurons_y):
            for x in range(neurons_x):
                activity = rearrange(normed_synapses[:, cnt])
                HM[y * self.width:(y+1) * self.width,
                   x * self.width:(x+1) * self.width, :] = activity
                cnt += 1
        if anim:
            return HM
        nc = np.max(np.absolute(HM))  #* normalize colorbar
        im = ax.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
        ax.set_title(f"Hebbian synapse activations of {self.name}.\n{self.ratio * 100:.2f}% of synapses have converged.")
        format_nice_plot()

        name = f"{self.path}/visualised_hebbian_synapses.png"
        plt.savefig(fname=name)
        plt.clf()

        #! cant be bothered with these anymore.
        # if self.figs.telesend and not self.aborted:
        #     telebot.send_img(
        #         name, msg=f"Hebbian Synapses of {self.name} just finished training. Here are they.")

    def to(self, device=None):
        """Send torch tensors to device"""
        if device is None: device = self.device
        self.hebb_synapses = self.hebb_synapses.to(device, copy=False)
        self._g_i          = self._g_i.to(device,          copy=False)
        self._indexer      = self._indexer.to(device,      copy=False)

    def detach(self):
        """Detach torch tensors from the computational graph and send to CPU"""
        self.hebb_synapses = self.hebb_synapses.detach().cpu()
        self._g_i          = self._g_i.detach().cpu()
        self._indexer      = self._indexer.detach().cpu()

    def patchify(self, x):
        """
        Takes an image (or N images) and returns a normalized tensor of patches.
        Equivalent to torch.nn.Unfold, but order of packing is different.
        nn.Unfold packs row-wise, this packs column-wise.
        Testing shows that this is faster than nn.Unfold. See testing_ground/unfolder_test.py.
        """
        patches = torch.zeros((x.size(0), self.patch_pixels, self.num_patches), device=self.device)                              #? (imgs, patch_pixels, patches)
        for row_idx in range(self._patches_pr_dim):
            row = x[:, row_idx: row_idx+self.width, :, :]                                                                        #? (imgs, width, dim,  channels)
            flattened = row.transpose(1, 2).reshape((x.size(0), -1))                                                             #? (imgs, dim * width * channels)
            for col_idx in range(self._patches_pr_dim):
                patch = flattened[:, col_idx * self.channels * self.width: (self.width + col_idx) * self.channels * self.width]  #? (imgs, patch_pixels)
                patches[:, :, row_idx * self._patches_pr_dim + col_idx] = patch

        norm = torch.linalg.norm(patches, dim=1, ord=self.p, keepdims=True)
        norm = torch.where(norm > 1e-7, norm, 1.)
        return patches / norm

    def train_unsupervised(self, x, epochs):
        """
        Training function for unsupervised learning.
        Finds new epoch count, sets up pbar, send x to device
        Train loop batches data, then patches it, then trains on the patches.
        The hebb_synapses are updated.
        Model is saved afterwards, if nothing has gone wrong.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before training"

        if self.trained:
            print(f"Model is already trained. Skipping training.")
            return

        self.time_of_training.append(datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))

        num_train = x.size(0)
        dim = x.size(1)
        assert num_train != 0, "No images to train on"

        #* number of patches along one dimension, formula from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        self._patches_pr_dim = (dim - (self.width - 1) - 1) // self.stride + 1
        self.num_patches = self._patches_pr_dim ** 2

        new_epochs = self.num_epochs + epochs
        pbar = tqdm(range(1, epochs * num_train // self.batch_size + 1))
        pbar.set_description(f"LL: Epoch 0/{new_epochs}. Conv: 0.00%. max|ds|: None")
        print(f"\nRunning unsupervised learning with {self.num_hidden} synapses.")
        print(f"Epochs: {epochs}, images: {num_train}, batch size: {self.batch_size}, patches pr image: {self.num_patches}\n")

        self.to(self.device)
        self.save_synapses_during_training(self.num_epochs)

        #* wrap as Tensor
        x = torch.Tensor(x).float().to(self.device)

        try:  #* try-except to catch keyboard interrupt for ending training early
            for epoch in range(epochs):

                #* parameterised learning rate
                eps = self.scheduler(epoch, epochs)

                #* shuffle data
                x = x[torch.randperm(num_train)]

                for batch in range(num_train // self.batch_size):  #* +1 to include last batch
                    #* exctract batch from x_train_flat
                    batch = x[batch * self.batch_size: (batch + 1) * self.batch_size]  #? (batch_size, dim, dim, channels)
                    patches = self.patchify(batch)  #* These are normalised            #? (batch_size, patch_pixels, patches)

                    for p_idx in range(self.num_patches):
                        patch = patches[:, :, p_idx]                                   #? (batch_size, patch_pixels)

                        #* I = <W,v> synaptic activation
                        tot_input = self.synaptic_activation(patch.t())                #? (K**2, batch_size)
                        #* find indices maximizing the synapse
                        _, indices = tot_input.topk(self.k, dim=0)                     #? (k, batch_size)
                        #* g(Q) learning activation function
                        self.learning_activation(indices)                              #? (K**2, batch_size)
                        #* Learning algorithm
                        xx = (self._g_i * tot_input).sum(dim=1)                        #? (K**2)
                        ds = (torch.matmul(self._g_i, patch) - \
                            xx.unsqueeze(1) * self.hebb_synapses)                      #? (K**2, W*W*C)
                        nc = max(ds.abs().max(), self._prec)
                        #* the actual update
                        self.hebb_synapses += eps * ds / nc                            #? (K**2, W*W*C)

                    ################################################
                    self._ds_evolution.append(ds.abs().max().item())
                    pbar.update(1)
                pbar.set_description(f"LL: Epoch {self.num_epochs}/{new_epochs}. Conv: {self.unconverged_neurons() * 100:.2f}%. max|ds|: {self._ds_evolution[-1]:.2f}")
                self.num_epochs += 1  #* only add epochs if we actually ran them
                if (self._save_every != 0) and (self.num_epochs % self._save_every == 0):
                    self.save_synapses_during_training(self.num_epochs)
                    self.save_model(verbose=False)
        except KeyboardInterrupt:  #* catch keyboard interrupt for ending training early
            print(f"KeyboardInterrupt. Aborting training early, after {epoch}/{epochs} epochs. Total epochs: {self.num_epochs}")
            self.aborted = True  #* set aborted to true so that it doesn't send a telegram message
            self.trained = True
            self.save_model()
        except Exception as e:  #* catch all other exceptions
            print(e)
            print("Something went wrong. Model is not saved.")
            self.aborted = True  #* set aborted to true so that it doesn't send a telegram message
            raise e
        else:  #* training completed without errors
            self.aborted = False
            self.trained = True
            self.save_model()
        finally:
            pbar.set_description(f"LL: Epoch {self.num_epochs}/{new_epochs}. Conv: {self.unconverged_neurons() * 100:.2f}%. max|ds|: {self._ds_evolution[-1]:.2f}")

    def learning_activation(self, indices):
        self._g_i *= 0  #* reset g_i
        best_ind, best_k_ind = indices[0], indices[self.k-1]
        self._g_i[best_ind,   self._indexer] = 1.0
        self._g_i[best_k_ind, self._indexer] = -self.delta

    def synaptic_activation(self, x):
        return (self.hebb_synapses.sign() * self.hebb_synapses.abs() ** (self.p - 1)).matmul(x)

    def forward(self, x):                #* (imgs, img_dim, img_dim,  channels)
        x = self.patchify(x)             #* (imgs, kernel_size * channels,  patches)
        x = self.synaptic_activation(x)  #* (imgs, hidden,  patches)
        return x

    def __call__(self, x):
        assert len(x.shape) == 4, "Input must be of shape (imgs, img_dim, img_dim,  channels)"
        # assert x.shape[1] == x.shape[2], "Input must be square, with shape (imgs, img_dim, img_dim,  channels)"
        return self.forward(x)

    def minmaxnorm(self, x):
        """
        Scales x so that values are between 0 and 1 column-wise.
        Only used when visualizing the synapses.
        """
        x_ = x - x.min(dim=0)[0]  #* set smallest values to 0
        maxes = x_.max(dim=0)[0]  #* find largest values
        maxes[(maxes < 1e-7).nonzero()] = 1  #* prevent divide by zero
        return x_ / maxes  #* scale largest values to 1

    def unconverged_neurons(self):
        """
        Returns the ratio of neurons whose norm is 1 and values are positive.
        """
        tol = 1e-5
        syn = self.hebb_synapses.detach().cpu()

        #* find neuron with all positive elements
        poss = (syn > 0).sum(dim=1) == syn.size(1)

        #* find neurons with norm 1
        norm = torch.linalg.norm(syn, dim=1, ord=self.p)
        converged = torch.abs(norm - 1) < tol

        #* Good neurons are those that are both all positive and have norm 1
        good_neurons = (poss & converged).sum().item()
        self.ratio = good_neurons / self.num_hidden
        self.good_model = self.ratio >= 0.1  #* More than 10% of neurons are good
        return self.ratio

    def save_model(self, verbose=True):
        """
        Saves the model to disk. All parameters are saved in a dictionary,
        including the synapse matrix.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before saving."
        self.print()  #* write model parameters to log file

        if verbose: self.detach()  #* assuming verbose only happens at the end of training
        with open(f"{self.path}/hebbian_parameters.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)
        if verbose: print(f"Model saved to {self.path}/hebbian_parameters.pkl")

    def save_synapses_during_training(self, epoch):
        """
        Saves the synapses to disk during training,
        so that we can see how they change over time in a gif.
        Only the synapse matrix is saved, in a subdir of the model dir.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before saving."

        path = f"{self.path}/saved_hebb_synapses"
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(f"{path}/epoch_{str(epoch).zfill(4)}.pkl", "wb") as f:
            pickle.dump(self.hebb_synapses.detach().cpu(), f)

    def load_model(self, name=None, verbose=True, device=None):
        """
        Loads the saved model parameters from disk.
        """
        if self._needs_parameter_loading:
            assert name is not None, "Empty model needs to be given a name to load parameters from."
            self.name = name

        with open(f"{self.path}/hebbian_parameters.pkl", "rb") as f:
            self.__dict__.update(pickle.load(f))
        if name is not None:
            self.name = name
            self.path = self.path.rsplit("/", 1)[0] + "/" + name
        if device is not None:
            self.device = device

        if verbose: print(f"Loaded parameters of {self.name} from file")
        self._needs_parameter_loading = False

        self.to(self.device)
        return self

    def purge(self, complete=False):
        """
        Removes the model directory from disk.
        """
        if complete:  #* delete entire folder
            shutil.rmtree(self.path)
        else:  #* delte all data except readable_parameters and the folder itself.
            try:
                os.remove(f"{self.path}/hebbian_parameters.pkl")
                os.remove(f"{self.path}/ds_evolve.png")
                os.remove(f"{self.path}/visualised_hebbian_synapses.png")
                shutil.rmtree(f"{self.path}/saved_hebb_synapses")
                shutil.rmtree(f"{self.path}/classifiers")
            except FileNotFoundError:
                pass

    def __str__(self):
        """
        Returns a string representation of the model.
        All parameters are printed in a nicely formatted table.
        """
        assert not self._needs_parameter_loading, "Model needs to be given parameters before printing."

        out = f"\nParameters of Hebbian layer of {self.name}:\n"
        out += "=" * len(out) + "\n"
        for key, val in self.__dict__.items():
            if key[0] == "_": continue  #* skip purely internal variables
            if key == "hebb_synapses":  #* skip the hebbian matrix
                val = f"synaptic matrix of shape {val.size()}"
            out += f"    {key:<25}: {val}\n"
        return out

    def print(self):
        """write parameters to file"""
        copy = sys.stdout  #* copy
        with open(f"{self.path}/readable_parameters.txt", "w") as f:
            sys.stdout = f  #* reassign
            print(self)  #* print to file
        sys.stdout = copy  #* put back copy
