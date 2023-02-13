from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from utils.utils import random_seed_generator as rsg
from copy import copy


class CustomDatasetClass:
    """
    Customized Dataset class. Uses most if the standard dataset class in torchvision, 
    but adds a x attribute, corresponding to data scaled by 1/255.0, so x \in (0, 1),
    and adds a y attribute, corresponding to one-hot encoded targets.

    Also adds a shuffle method, which shuffles the data and targets, and x and y. 
    """
    def __init__(self, Data, args, train):
        self.original = Data
        self.train = train
        self.args = args

        if args.dataset == "MNIST":  # add colour channel for MNIST
            Data.data = Data.data[:, :, :, None]
        else:  # convert cifar data to torch tensors
            Data.data = torch.from_numpy(Data.data)
        Data.targets = torch.Tensor(Data.targets).long()

        self.data = copy(self.original.data)
        self.x = copy(self.original.data  / 255.0)
        self.targets = copy(self.original.targets)
        self.y = copy(one_hot(self.original.targets))
        self.classes = self.original.classes
        self.num_classes = len(self.classes)

        self.generator = torch.Generator()
        self.generator.manual_seed(rsg(seed=args.seed))

    def reset(self):
        self.data = copy(self.original.data)
        self.x = copy(self.original.data  / 255.0)
        self.targets = copy(self.original.targets)
        self.y = copy(one_hot(self.original.targets))

        self.generator.manual_seed(rsg(seed=self.args.seed))

    def set_chw(self, set=True):
        pass
        """
        Set the ordering of the data to channels first (NCHW).
        This is legacy, now the relevant models are set to channels first ordering themselves.
        """
        # if set:
        #     self.x = self.x.permute(0, 3, 1, 2)  #* channels first ordering
        # else:  #! this is left as a warning for the future. This else statement made me debug for 5 hours. 
        #     self.x = self.x.permute(0, 2, 3, 1)  #* channels last ordering

    def shuffle(self):
        """Shuffle the data and targets, and x and y."""
        perm = torch.randperm(self.x.shape[0], generator=self.generator)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.data = self.data[perm]
        self.targets = self.targets[perm]

    def drop_wrong(self, model, batch_size, certainty=0):
        """ Drop all wrong predictions from the dataset """
        right = []
        for bidx in range(0, len(self), batch_size):
            x = self.x[bidx : bidx + batch_size]
            y = self.targets[bidx : bidx + batch_size]
            with torch.no_grad():
                c, pred = model(x).max(dim=1, keepdim=False)
                if certainty != 0:
                    right.append(torch.where((pred.eq(y)) & (c.exp() > certainty))[0] + bidx)
                else:
                    right.append(torch.where(pred.eq(y))[0] + bidx)
        right = torch.cat(right)
        self.x = self.x[right]
        self.y = self.y[right]
        self.data = self.data[right]
        self.targets = self.targets[right]

    def reduce(self, count):
        if count != 0:
            self.x = self.x[:count]
            self.y = self.y[:count]
            self.data = self.data[:count]
            self.targets = self.targets[:count]

    def to(self, device):
        """Send all tensors to device"""
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)

    def detach(self):
        """Detach all tensors from the graph and send to cpu"""
        self.x = self.x.detach().cpu()
        self.y = self.y.detach().cpu()
        self.data = self.data.detach().cpu()
        self.targets = self.targets.detach().cpu()

    def get(self, index, xy=False):
        if xy:
            return self.x[index], self.y[index]
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.x.size(0)
    
    def __repr__(self):
        return self.original.__repr__()


def load_data(args, train=True, path="./Data"):
    assert args.dataset in ("MNIST", "CIFAR")
    if hasattr(args, "path"):
        path = args.path

    if args.dataset == "MNIST":
        dataloader = datasets.MNIST
        transformer = transforms.Compose(
            [transforms.Normalize((0.1307), (0.3081))])  # subtract mean and divide by std of data
    else:
        dataloader = datasets.CIFAR10
        transformer = transforms.Compose([])

    Data = dataloader(root=path, train=train,
                      transform=transformer, download=True)
    
    return CustomDatasetClass(Data, args, train)

def load_datas(args):
    return load_data(args, train=True), load_data(args, train=False)


def one_hot(y):
    """"
    Makes target-list 'one-hot' (self-indexing)

    Input: y, 1d-tensor-like
    Returns: Y, 2d-tensor

    Examples:
    >>> one_hot([0, 3, 1, 1]) = 
                [[1, 0, 0, 0],
                 [0, 0, 0, 1], 
                 [0, 1, 0, 0], 
                 [0, 1, 0, 0]]
    """
    Y = torch.zeros((y.size(0), y.max().item() + 1), dtype=y.dtype)
    Y[torch.arange(y.size(0)), y.numpy()] = 1
    return Y

class Foo:
    def __init__(self):
        self.dataset = "CIFAR"
        self.seed = 1

if __name__ == '__main__':
    A = Foo()
    train, test = load_datas(A)

    fig, axs = plt.subplots(nrows=3, ncols=5)
    axs = axs.flatten()
    for i in range(len(axs)):
        x, y = train.get(torch.randint(len(train), (1,)).squeeze())
        axs[i].imshow(x)
        axs[i].set_title(train.classes[y])
        axs[i].axis("off")
    plt.savefig("image_examples.png")
