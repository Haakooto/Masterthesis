"""
File with functions for setting up grid search
Each function returns a dictionary of commands to run
Do not change or remove any functions, as they represent the different experiments I ran in the course of my thesis
"""

import sys

def dpkw():
    learning_rate = 1e-4
    batch_sizes = 10000
    num_epochs = 750
    K = 20

    deltas = [0.2, 0.4]
    ps = [2,3,4,5,6]
    ks = [3,4,5,6,7]
    widths = [4, 8]

    base = f"python train_models.py --grid-search --mode local-learning --dataset CIFAR "
    base += f"--eps0 {learning_rate} --batch-size {batch_sizes} --num-epochs {num_epochs} --K {K} "
    base += f"--save-every 50 --seed -1 --device cuda "

    for delta in deltas:
        for width in widths:
            for p in ps:
                for k in ks:
                    command = base + f"--delta {delta} --width {width} --p {p} --k {k} "
                    name = "Alice"
                    name += "Strong" if delta == 0.4 else "Weak"
                    name += "Thin" if width == 4 else "Wide"
                    name += f"{p}{k}"
                    command += f" --name {name}"
                    print(command)

def eps0_withp3k7():
    base = f"python train_models.py --grid-search --mode local-learning "
    base += f"--batch-size 2000 --num-epochs 750 "
    base += f"--delta 0.4 --p 3 --k 7 "

    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    for i, learning_rate in enumerate(learning_rates):
        command = base + f"--eps0 {learning_rate} "
        name = f"lr_search{i}"
        command += f"--name {name}"
        print(command)

def monkey_brains(make=True):
    func = "monkey_brains:"
    base = "python train_models.py --grid-search --mode local-learning --batch-size 1000 --num-epochs 50 "
    base += "--seed -1 --save-every 10 --figs notelesend "

    eps0s = [75, 100, 125]
    lr_types = ["linear", "cosine", "exp"]
    eps_decays = {"linear": [0,], "cosine": [0,], "exp": [2.5, 3.25, 4]}
    ks = [2, 3, 4, 5]
    ps = [2, 3, 4, 5]
    Ks = [14, 20]
    deltas = [0.2, 0.4]

    family = {}

    out = ""

    for eps0 in eps0s:
        for lr_type in lr_types:
            for edi, eps_decay in enumerate(eps_decays[lr_type]):
                for k in ks:
                    for p in ps:
                        for K in Ks:
                            for delta in deltas:
                                command = base + f"--eps0 {eps0}e-5 --lr-type {lr_type} "
                                command += f"--eps-decay {eps_decay} --k {k} --p {p} --K {K} --delta {delta} "
                                name = f"MB_{lr_type}{str(eps0).zfill(3)}_{edi}_"
                                name += "Big" if K == 20 else "Small"
                                name += "Strong" if delta == 0.4 else "Weak"
                                name += f"{k}{p}"
                                command += f"--name {name}"
                                family[name] = command
                                out += f"\t{command}\n"
    if make:
        write_to_make(func, out)
    else:
        return family

def classifiers(Hebb_name="Carmen"):
    eps0s = [0.001, 0.004, 0.008]
    gammas = [0.1, 0.5, 0.9]
    steps = [2, 5, 10]
    # powers = [16, 18, 20, 22, 24, 26, 28, 30]
    powers = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    biases = [False,]
    types = ["linear", "cosine", "exp", "step"]

    base = "python train_models.py --grid-search --device cuda:0 --mode classify --batch-size 1000 --num-epochs 100 "
    family = {}

    for bias in biases:
        for power in powers:
            for type in types:
                for eps0 in eps0s:
                    command = base + f"{'--use-bias ' if bias else ''}--power {power}"
                    command += f" --scheduler {eps0} {type}"
                    hebb_name = f" --name {Hebb_name} "
                    name = f"{'B' if bias else 'N'}{power}_{type}{int(eps0*1000)}"

                    if type == "step":
                        for gamma in gammas:
                            for step in steps:
                                type_command = command + f" {step} {gamma}"
                                type_name = name + f"_{step}_{int(gamma*10)}"
                                type_command += f"{hebb_name}{type_name}"
                                family[type_name] = type_command

                    elif type == "exp":
                        for gamma in gammas:
                            type_command = command + f" {gamma}"
                            type_name = name + f"_{int(gamma*10)}"
                            type_command += f"{hebb_name}{type_name}"
                            family[type_name] = type_command
                    else:
                        command += f"{hebb_name}{name}"
                        family[name] = command
    return family
    # print(family)

def ProjectMercuryHebbian():
    """Project Mercury Hebbians. K=10 is for all, and scheduler is not changed."""
    base = "--grid-search --save-all --save-every 2 --num-epochs 50 --batch-size 1000 --seed 1"
    base += " --mode local-learning --dataset CIFAR"
    base += " --K 10 --scheduler 0.001 cosine"

    deltas = [0.1, 0.2]
    widths = [4, 8]
    ps = [2, 3, 4, 5, 6]
    ks = [2, 3, 4, 5, 6]

    family = {}
    for delta in deltas:
        for width in widths:
            for p in ps:
                for k in ks:
                    command = base + f" --delta {delta} --width {width} --p {p} --k {k} "
                    name = "Feeble" if delta == 0.1 else "Weak"
                    name += "Thin" if width == 4 else "Wide"
                    name += f"{p}{k}"
                    command += f" --name {name}"
                    family[name] = command
    return family

def ProjectMercuryClassifiers(Hebb_name):
    """Project Mercury Classifiers. only power and eps0 are changed."""
    base = "--grid-search --seed 1 --mode classify --dataset CIFAR --batch-size 1000 --num-epochs 100 --maxpool"

    powers = [2, 4, 6, 8, 10, 12]
    eps0s = [0.0008, 0.001, 0.002]

    family = {}
    for power in powers:
        for eps0 in eps0s:
            command = base + f" --power {power} --scheduler {eps0} cosine"
            name = f"P{str(eps0)[-1]}{power}"
            command += f" --name {Hebb_name} {name}"
            family[name] = command
    return family

def ProjectMercuryClassifiers_restricted(Hebb_name):
    """A second set of Project Mercury Classifiers. Also varies maxPool kernel size, but for fewer powers"""
    base = "--grid-search --seed 1 --mode classify --dataset CIFAR --batch-size 1000 --num-epochs 100 --maxpool-stride 2"

    powers = [8, 10, 12]
    eps0s = [0.002,]
    kernels = [5, 7, 9, 13]

    family = {}
    for power in powers:
        for eps0 in eps0s:
            for kernel in kernels:
                command = base + f" --power {power} --scheduler {eps0} cosine --maxpool-kernel {kernel}"
                name = f"P{str(eps0)[-1]}{power}_{kernel}"
                command += f" --name {Hebb_name} {name}"
                family[name] = command
    return family

def ProjectMercuryBP():
    """Project Mercury BackPropers."""
    base = "--mode endtoend --seed 1 --batch-size 1000 --num-epochs 100 --grid-search"

    widths = [4, 8]
    Ks = [10, 14, 20]
    powers = [2, 4, 6, 8, 10, 12, 14]
    eps0s = [0.0008, 0.001, 0.002]
    maxpool_kernels = [1, 5, 7, 9, 11, 13]

    family = {}
    for power in powers:
        for eps0 in eps0s:
            for K in Ks:
                for width in widths:
                    for kernel in maxpool_kernels:
                        if kernel == 1:
                            stride = 1
                        else:
                            stride = 2
                        command = base + f" --power {power} --scheduler {eps0} cosine --maxpool-kernel {kernel} --maxpool-stride {stride} --K {K} --width {width}"
                        name = f"BP_K{K}_w{width}_e{str(eps0)[-1]}_p{power}_x{kernel}"
                        command += f" --name {name}"
                        family[name] = command
    return family

def Activation_Hebb():
    base = "--mode local-learning --num-epochs 30 --save-all --seed -1 --grid-search --delta 0.1"

    Ks = [10, 20]
    widths = [4, 8, 12]
    ps = [2,3,4,5,6]
    ks = [2,3,4,5,6]

    family = {}
    for K in Ks:
        for width in widths:
            for p in ps:
                for k in ks:
                    command = base + f" --K {K} --width {width} --p {p} --k {k}"
                    name = f"K{K}_w{width}_{p}{k}"
                    command += f" --name {name}"
                    family[name] = command
    return family, "Activate"

def Activation_Classifiers(Hebb_name):
    base = "--mode classify --num-epochs 100 --save-all --seed -1 --grid-search --scheduler 2e-3 cosine --maxpool-stride 2"

    powers = [4, 6, 8, 10, 12]
    activations = ["relu", "elu", "gelu", "exp", "rexp"]
    mxpools = [5, 7, 9, 11, 13]

    family = {}
    for mxpool in mxpools:
        for activation in activations:
            if activation in ("exp", "rexp"):
                command = base + f" --activate {activation} --maxpool-kernel {mxpool}"
                name = f"MX{str(mxpool).zfill(2)}_{activation}"
                command += f" --name {Hebb_name} {name}"
                family[name] = command

            else:
                for power in powers:
                    command = base + f" --power {power} --activate {activation} --maxpool-kernel {mxpool}"
                    name = f"MX{str(mxpool).zfill(2)}_{activation}{str(power).zfill(2)}"
                    command += f" --name {Hebb_name} {name}"
                    family[name] = command
    return family, "Activate"

def Activation_BP():
    base = "--mode endtoend --family ActivateBP --num-epochs 70 --save-all --seed -1 --grid-search --maxpool-stride 2"

    powers = [2, 4, 6, 8, 10, 12]
    activations = ["relu", "elu", "gelu", "exp", "rexp"]
    mxpools = [5, 7, 9, 11, 13]
    widths = [4, 8, 12]
    Ks = [10, 20]

    family = {}
    for K in Ks:
        for width in widths:
            for mxpool in mxpools:
                for activation in activations:
                    if activation in ("exp", "rexp"):
                        command = base + f" --activate {activation} --maxpool-kernel {mxpool} --K {K} --width {width}"
                        name = f'BP_K{K}_w{str(width).zfill(2)}_MX{str(mxpool).zfill(2)}_{activation}'
                        command += f" --name {name}"
                        family[name] = command

                    else:
                        for power in powers:
                            command = base + f" --power {power} --activate {activation} --maxpool-kernel {mxpool} --K {K} --width {width}"
                            name = f'BP_K{K}_w{str(width).zfill(2)}_MX{str(mxpool).zfill(2)}_{activation}{str(power).zfill(2)}'
                            command += f" --name {name}"
                            family[name] = command
    return family, "ActivateBP"

def write_to_make(func, cmds):
    """Dumps the commands to a makefile. Legacy"""
    og = sys.stdout
    with open(f"Makefile", "a") as f:
        sys.stdout = f
        print(func)
        print(cmds)
        # for cmd in cmds:
        #     print("\t" + cmd)
    sys.stdout = og

def main():
    """Only used when dumping to makefile. Legacy. Use run_the_grid.py instead."""
    # dpkw()
    # eps0_withp3k7()
    # monkey_brains()
    # Carmens()

if __name__ == "__main__":
    print("Dont do this. Use run_the_grid.py instead.")
    main()


