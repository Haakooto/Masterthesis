from collections import defaultdict
import torch
import glob
import ast

"""
Path utils
These might look daunting, but they are here for your convenience!
They make sure that the models are saved in the right place, and that you can easily load them.
Let me explain:
    All layers are saved in the folder Models/
    Because there are many, I want to save them in subfolders, called a family.
    Each hebbian layer is a subfolder in the family, with a pickled file containing the model. There are some other related files as well.
    Each classification layer is a subfolder in classifier-folder fo the hebbian layer, with a pickled file containing the model. There are some other related files as well.
    The family path is hard-coded below.
    The hebb_path gives path to local-learning layers. If name is given, to the specified layer folder, otherwise to the family folder.
    The clas_path gives path to classification layers. The local-learning layer it is based on must be given.
        As with hebb_path, if name is given, you get the specified classification layer folder, otherwise the classifiers-folder to the local-learning layer.
    The existing_models function gives list of all layers within the family. If name is not given, it gives all local-learning layers.
        If name is given, it gives all classification layers under that local-learning layer.
        The last if statement is to avoid giving the family folder as a model.
"""
famy_path = "Activate"
hebb_path = lambda name=False, fam=famy_path: f"./Models/{fam}/{name if name else ''}"
clas_path = lambda hebb_name=False, name=False, fam=famy_path: f"{hebb_path(name=hebb_name, fam=fam)}/classifiers/{name if name else ''}"
existing_models = lambda name=False, fam=famy_path: sorted([path.split("/")[-1] for path in glob.glob((clas_path(hebb_name=name, fam=fam) if name else hebb_path(fam=fam)) + "*") if "." not in path[1:]])

path_utils_pack = [famy_path, hebb_path, clas_path, existing_models]

final_figure_path = "/home/users/haakooto/local_learning_robustness/tex/figures"

#* Backward compatibility fixes
try:
    import utils.schedule_selection as ss
    LocalScheduler = ss.LocalScheduler
    ClassifyScheduler = ss.TorchScheduler
    TorchScheduler = ss.TorchScheduler
except ImportError:
    pass

def pick_device(device: str) -> torch.device:
    """
    Picks the device to use for training.
    Finds the first available GPU if device is "cuda" and if no GPU is available, uses CPU.
    """
    override = False
    if device == "cpu":
        return torch.device("cpu")
    elif ":" in device:  #* specific GPU is specified
        return torch.device(device)
    elif device == "nocpu":
        override = True
        device = "cuda"
    if device == "cuda":
        for d in list(range(torch.cuda.device_count())):
            if not torch.cuda.memory_usage(d):
                return torch.device(f"cuda:{d}")
        if not override:
            cpu = input("No free GPU found. Use CPU? [y/n] ")
            if cpu == "y":
                return torch.device("cpu")
        raise RuntimeError("No free cuda devices found")

class FigArgs:
    """
    Collects all arguments for plotting in one place.
    Right now, the only argument is telesend,
    and only used by LocalLearning for the hebbian synapses
    """
    def __init__(self, args):
        self.telesend = False

        while len(args) > 0:
            arg = args.pop(0)
            if arg == "telesend":
                self.telesend = True
            else:
                raise ValueError("Unknown figure argument: {}".format(arg))

    def __repr__(self) -> str:
        return f"FigArgs(telesend={self.telesend})"


def random_seed_generator(seed=-1, n=16):
    """
    Generates a good initial seed for a random number generator
    A good random seed has a good balance between 0s and 1s in the binary representation
    This is because the random number generator is a linear congruential generator  # https://en.wikipedia.org/wiki/Linear_congruential_generator
    https://stackoverflow.com/questions/29916065/how-to-choose-a-seed-for-numpy-random
    (Dont believe my tabnine, its's literally making stuff up)

    Parameters
    ----------
    n : int
        Number of 0s and 1s in the binary representation of the seed
    seed : int
        Seed for the random number generator. Default is -1, in which case the seed is not set manually
    Returns
    -------
    seed : int
        The generated seed
    """
    from numpy.random import default_rng as rng
    gen = rng(seed if seed != -1 else None)
    #* make (n ± rand(-5, 0)) 0s and (n ± rand(-2, 2)) 1s in a list
    s = ["0"] * (n + gen.integers(6) - 5) + ["1"] * (n + gen.integers(5) - 2)
    gen.shuffle(s)
    s = "".join(s)
    return int(s, base=2)  #* convert binary number to base 10


def get_directly_implemented_funcs(module):
    """
    Returns the functions implemented in the given module.
    The functions has to be directly implemented (not imported),
    and declared using def.
    The returned dict has the name of the functions as keys,
    and reference to them as values.
    """
    s = open(f"./{module.__name__.replace('.', '/')}.py").read()
    flist = {}
    for f in ast.parse(s).body:
        if isinstance(f, ast.FunctionDef):
            flist[f.name] = eval("module." + f.name)
    return flist


def Get_all_model_info():
    #! Not sure what I was doing here
    Models = NestableDottableDefaultDict()
    generations = [path.split("/")[-1] for path in glob.glob("./Models/*")]
    print(generations)
    # Models.generations = generations
    # print(hebb_path(fam_path="ProjectMercury"))
    # print(existing_models())
    for generation in generations:
        for model in existing_models(fam_path=generation):
            Models["generations"][generation][model] = {}
        # print(existing_models(fam_path=generation))
        # input()
    
    print(Models)


def minmaxnorm(x):
        """
        Scales x so that values are between 0 and 1 column-wise.
        Only used when visualizing the synapses.
        """
        x_ = x - x.min(dim=0)[0]  #* set smallest values to 0
        maxes = x_.max(dim=0)[0]  #* find largest values
        maxes[(maxes < 1e-7).nonzero()] = 1  #* prevent divide by zero
        return x_ / maxes  #* scale largest values to 1

def main():
    args = train_argparser()
    print(args)

    for _ in range(20):
        seed = random_seed_generator()
        print(f"Seed: {seed:<12}", end="  ")
        print(f"0-count: {bin(seed).count('0')-1:>2}", end="   ")
        print(f"1-count: {bin(seed).count('1')  :>2}")

if __name__ == "__main__":
    main()

