import argparse
import torch
from utils.utils import FigArgs, pick_device, random_seed_generator
from utils.schedule_selection import LocalScheduler, TorchScheduler, AttackStrength
import utils.utils as utils
famy_path, hebb_path, clas_path, existing_models = utils.path_utils_pack


def attack_argparser(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument  # shorthand

    add_arg("--name",       type=str, nargs="+",       help="name of the model. Must be given. For classification, 2 names are needed, separated by space", required=True)
    add_arg("--family",     type=str, default="None",  help="Family name of model. If not given, defaults to 'famy_path' set in utils/utils.py")
    add_arg('--dataset',    type=str, default='CIFAR', help="dataset to use. Default: CIFAR", choices=['CIFAR', 'MNIST'])
    add_arg('--batch-size', type=int, default=1000,    help="batch size to use. Default: 1000")
    add_arg('--examples',   type=int, default=0,       help="Number of datapoints to attack on. If not given, use entire dataset. Default: 0")
    add_arg("--repeats",    type=int, default=1,       help="number of times to repeat the attack. Default: 1")
    add_arg('--device',     type=str, default="cuda",  help="torch device to use. Default: cuda")
    add_arg("--seed",       type=int, default=1,       help="random seed to use for seed generation. Defaults to 1. Set to -1 to use a random seed")
    add_arg("--figs",       nargs="+",default=[],      help="Figure arguments. Can be any subset of ['notelesend']. Telesend will send the saved image to my phone.")
    add_arg("--eps",        nargs="+",default=[],      help="Determines the epsilon values for the attack. Default: []")
    add_arg("--rerun",      action="store_true",       help="Whether to rerun the attack in the model Default: False")
    add_arg("--no-hist",    action="store_true",       help="Whether to plot histogram. Can ble slow. Default: False")
    add_arg("--family-attack",action="store_true",     help="Whether to attack the family of models. Default: False")
    add_arg("--certainty",  type=float,default=0,      help="certainty condition for cherry-picking. Default: 0")
    add_arg("--cutoff",     type=float,default=0.01,   help="Accuracy cutoff when plotting range. Default: 0")
    add_arg("--pad",        type=float,default=0.10,   help="Padding to each end when fitting, in percent of number of data points. Default: 0")
    add_arg('--attack',     type=str, default="print", help="section of experiment to run. Default: print",
               choices=["print",    #* Just print the parameters of the model
                        "nothing",  #* Dont perturbe input
                        "random",   #* Random noise
                        "rotate",   #* rotate inputs
                        "occlude",  #* occlude inputs
                        "fgsm",     #* Fast Gradient Sign Method
                        "pdg",      #* Projected Gradient Descent
                        "deepfool", #* DeepFool
                        "cw",       #* Carlini & Wagner
                        "jsma",     #* Jacobian-based Saliency Map Attack
                        ])
    attack_args = parser.parse_args(args)
    attack_args.BP = len(attack_args.name) == 1

    #* These args are changed from the default values
    attack_args.figs = FigArgs(attack_args.figs)
    attack_args.seed = random_seed_generator(seed=attack_args.seed) if attack_args.seed < 1e4 else attack_args.seed
    attack_args.device = pick_device(attack_args.device)
    attack_args.eps = AttackStrength(attack_args.eps)
    if attack_args.family == "None":
        attack_args.family = famy_path

    return attack_args

def simple_attack_argparser(args=None) -> argparse.Namespace:
    """ old frameworks are too convoluted, so a complete reworking of attacking is needed. This is the argparser for the simpler framework. """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument  # shorthand

    add_arg("--name", type=str, nargs="+")
    add_arg("--family", type=str, default="None", required=True)
    add_arg("--force", action="store_true")
    add_arg("--plot", action="store_true")
    add_arg("--no-hist", action="store_true")
    add_arg("--batch-size", type=int, default=1000)
    add_arg('--dataset', type=str, default='CIFAR', choices=['CIFAR', 'MNIST'])
    add_arg('--device', type=str, default="cuda")
    add_arg("--seed", type=int, default=1)
    add_arg("--attack", type=str, choices=["fgsm", "pgd"])
    add_arg("--certainty", type=float, default=0)
    add_arg("--cutoff", type=float, default=0.00)
    add_arg("--pad", type=float, default=0.10)
    add_arg("--size", type=float, default=0.12)  #* size of projection region. Default: ~10 rgb values
    add_arg("--step", type=float, default=0.001)  #* step size for pgd. Default: 0.1275 rgb values

    parsed = parser.parse_args(args)
    parsed.family = famy_path if parsed.family == "None" else parsed.family
    parsed.BP = parsed.family[-2:] == "BP"

    if parsed.certainty > 1:  #* Samples are given with certainty param above 1.
        parsed.certainty = int(parsed.certainty)

    if parsed.name is None:  #* No specific model given, so attack the entire family
        parsed.fam_attack = True 
        # path = hebb_path(name=existing_models(fam=parsed.family)[0], fam=parsed.family)
        # try:  #* Need to determine if BP or not, try to load BP_params for the first member of family
        #     open(f"{path}/BP_parameters.pkl", "rb")
        # except FileNotFoundError:
        #     parsed.BP = False  #* If BP_params not found, then not BP
        # else:
        #     parsed.BP = True  #* If BP_params found, then BP
    else:  #* Specific model given, so attack that model
        parsed.fam_attack = False
        parsed.BP = len(parsed.name) == 1  #* BP determined by number of names
        if parsed.BP:
            parsed.name = parsed.name[0]
        else:
            parsed.hebb_name = parsed.name[0]
            parsed.name = parsed.name[1]
        
    parsed.seed = random_seed_generator(seed=parsed.seed) if parsed.seed < 1e4 else parsed.seed
    parsed.device = pick_device(parsed.device)
    return parsed

def train_argparser(args=None) -> argparse.Namespace:
    """
    Argparser for training a model.
    Some over-arching arguments are defined here, and then the mode specific arguments are added.
    """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument  # shorthand

    add_arg("--grid-search",   action="store_true", help="If set, will store parameters in an external dataframe together with evaluation of model. Default: False")
    add_arg("--force-retrain", action="store_true", help="if model is already trained, the synapses will not be retrained. This overrides this")
    add_arg("--train-more",    action="store_true", help="Continue training with more epochs. Default: False")
    add_arg("--save-all",      action="store_true", help="Save all models, not just the good ones. Default: False")
    add_arg("--name",     type=str, nargs="+",      help="name of the model. Must be given. For classification, 2 names are needed, separated by space", required=True)
    add_arg("--family",   type=str, default="None", help="Family name of model. If not given, defaults to 'famy_path' set in utils/utils.py")
    add_arg('--dataset',  type=str, default='CIFAR',help="dataset to use. Default: CIFAR", choices=['CIFAR', 'MNIST'])
    add_arg('--device',   type=str, default="cuda", help="torch device to use. Default: cuda")
    add_arg("--seed",     type=int, default=1,      help="random seed to use for seed generation. Defaults to 1. Set to -1 to use a random seed")
    add_arg("--figs",     nargs="+",default=[],     help="Figure arguments. Can be any subset of ['notelesend']. Telesend will send the saved image to my phone.")
    add_arg("--scheduler",nargs="+",default=[],     help="learning rate params. First is eps0, second is type. Rest are type specific params. Default: eps0=1e-3, type=linear")
    add_arg('--mode',     type=str, default="print",help="section of experiment to run. Default: print",
                choices=["local-learning",  #* train layer 1 of local model
                         "classify",        #* train layer 2 of local model
                         "endtoend",        #* train a non-local BP-ed model
                         "print",           #* print parameters of a model
                         "animate-weights", #* make gif of hebbian weight evolution
                         "rename",          #* rename a model. The new name is taken by prompting at runtime
                         ])
    shared_args, rest_args = parser.parse_known_args(args)

    #* These args are changed from the default values
    shared_args.figs = FigArgs(shared_args.figs)
    shared_args.seed = random_seed_generator(seed=shared_args.seed) if shared_args.seed < 1e4 else shared_args.seed
    shared_args.channels = 3 if shared_args.dataset == "CIFAR" else 1
    shared_args.device = pick_device(shared_args.device)
    if shared_args.family == "None":
        shared_args.family = famy_path


    if shared_args.mode in ("local-learning", "animate-weights"):
        assert len(shared_args.name) == 1, "Only one model name can be given for local learning"
        shared_args.name = shared_args.name[0]
        shared_args.scheduler = LocalScheduler(shared_args.scheduler)
        specific_args = local_argparser(rest_args)

    elif shared_args.mode in ("classify"):
        assert len(shared_args.name) == 2, "Classification mode requires two model names"
        shared_args.hebb_name = shared_args.name[0]
        shared_args.name = shared_args.name[1]
        shared_args.scheduler = TorchScheduler(shared_args.scheduler)
        specific_args = classification_argparser(rest_args)

    elif shared_args.mode in ("endtoend"):
        assert len(shared_args.name) == 1, "Only one model name can be given for end-to-end training"
        shared_args.name = shared_args.name[0]
        shared_args.scheduler = TorchScheduler(shared_args.scheduler)
        specific_args = end_to_end_argparser(rest_args)

    elif shared_args.mode in ("print", "rename", "dummy"):
        specific_args = argparse.Namespace()  #* no extra args are needed for these

    #* merge shared and specific args
    return argparse.Namespace(**vars(shared_args), **vars(specific_args))


def local_argparser(args) -> argparse.Namespace:
    """
    Picks the arguments for local learning
    """
    local_parser = argparse.ArgumentParser()
    add_arg = local_parser.add_argument  # shorthand

    add_arg("--mu",         type=float, default=0.0,   help="mean of gaussian for initialization of synapses. Default: 0.0")
    add_arg("--prec",       type=float, default=1e-30, help="precision for numerical stability. Default: 1e-30")
    add_arg("--delta",      type=float, default=0.2,   help="strength of anti-hebbian learning. Default: 0.2")
    add_arg("--sigma",      type=float, default=1,     help="std of gaussian for initialization of synapses. Default: 1")
    add_arg("--batch-size", type=int,   default=1000,  help="batch size. Default: 12500")
    add_arg("--num-epochs", type=int,   default=50,    help="number of epochs. Default: 1000")
    add_arg("--save-every", type=int,   default=10,    help="save model every n epochs. If n is 0, none is saved. Default: 50.")
    add_arg("--stride",     type=int,   default=1,     help="stride of convolution. Defaults to 1")
    add_arg("--width",      type=int,   default=4,     help="width of convolution. Defaults to 8")
    add_arg("--p",          type=int,   default=2,     help="Lebesgue norm of the weights. Default: 2")
    add_arg("--k",          type=int,   default=2,     help="ranking parameter, must be interger geq 2. Defaults: 2")
    add_arg("--K",          type=int,   default=20,    help="square root of hidden units. Default: 20")

    return local_parser.parse_args(args)


def classification_argparser(args) -> argparse.Namespace:
    """
    Picks the arguments for classification
    """
    classification_parser = argparse.ArgumentParser()
    add_arg = classification_parser.add_argument  # shorthand

    add_arg("--power",         type=float, default=4.0,   help="Power in SteepRElU. Default: 4.0")
    add_arg("--num-epochs",    type=int,   default=100,   help="Number of epochs. Default: 300")
    add_arg("--batch-size",    type=int,   default=1000,  help="Batch size. Default: 1000")
    add_arg("--maxpool-kernel",type=int,   default=11,    help="maxpool kernel size. Default: 1")
    add_arg("--maxpool-stride",type=int,   default=2,     help="maxpool stride. Default: 1")
    add_arg("--use-bias",      action="store_true",       help="Use bias in classification layer. Default: False")
    add_arg("--no-test",       action="store_true",       help="Wether to evaluate accuracy on test set during training. Default: True")
    add_arg("--activate",      type=str,   default="relu",help="Activation function to use. Default: relu",
                               choices=["relu", "gelu", "elu", "exp", "rexp"],)

    return classification_parser.parse_args(args)


def end_to_end_argparser(args) -> argparse.Namespace:
    """
    Picks the arguments for end-to-end learning
    """
    EtoE = argparse.ArgumentParser()
    add_arg = EtoE.add_argument  # shorthand

    add_arg("--maxpool-stride",type=int,   default=2,     help="maxpool stride. Default: 1")
    add_arg("--maxpool-kernel",type=int,   default=11,    help="maxpool kernel size. Default: 1")
    add_arg("--num-epochs",    type=int,   default=50,    help="number of epochs. Default: 50")
    add_arg("--batch-size",    type=int,   default=1000,  help="batch size. Default: 1000")
    add_arg("--stride",        type=int,   default=1,     help="stride of convolution. Defaults to 1")
    add_arg("--width",         type=int,   default=4,     help="width of convolution. Defaults to 4")
    add_arg("--K",             type=int,   default=20,    help="square root of hidden units. Default: 20")
    add_arg("--power",         type=float, default=4.0,   help="Power in SteepRElU. Default: 4.0")
    add_arg("--no-test",       action="store_true",       help="Wether to evaluate accuracy on test set during training. Default: True")
    add_arg("--use-bias",      action="store_true",       help="Use bias in Conv2d and Linear. Default: False")
    add_arg("--activate",      type=str,   default="relu",help="Activation function to use. Default: relu",
                               choices=["relu", "gelu", "elu", "exp", "rexp"],)

    return EtoE.parse_args(args)

