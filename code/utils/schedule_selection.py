import torch
from utils.CustomObjects import DefPopList

class LocalScheduler:
    """
    Scheduler for local learning rate. All are implemented by me
    """

    def __init__(self, args):
        self.pi = torch.tensor(torch.pi)
        if args == []:
            args = ["1e-3", "cosine"]  # * Default values

        if len(args) < 2:
            raise ValueError("Not enough arguments given to LocalScheduler")
        self.eps0 = float(args[0])
        self.lr_type = args[1]

        # * Extract lr_type specific arguments
        if self.lr_type == "exp":
            if len(args) != 3:
                raise ValueError("Exponential decay requires eps_decay")
            self.eps_decay = torch.tensor(float(args[2]))
        if self.lr_type == "step":
            if len(args) != 4:
                raise ValueError("Exponential decay requires eps_decay")
            self.gamma = torch.tensor(float(args[2]))
            self.step_size = torch.tensor(float(args[3]))
        else:
            if len(args) != 2:
                raise ValueError("Unknown arguments given to LocalScheduler")

        implemented_funcs = {"linear": (self.linear, True),
                             "cosine": (self.cosine, True),
                             "exp":    (self.exp,    True),
                             "custom": (self.custom, True),
                             "step":   (self.step,   False),
                             }
        if self.lr_type in implemented_funcs:
            self.lr_at, self.ratio = implemented_funcs[self.lr_type]
        else:
            raise ValueError("Unknown lr type")

    def __call__(self, x, X):
        if self.ratio:
            return self.eps0 * self.lr_at(x / X)
        return self.eps0 * self.lr_at(x)

    def linear(self, t):
        return (1 - t)

    def exp(self, t):
        return torch.exp(-self.eps_decay * t)

    def cosine(self, t):
        return 0.5 * (1 + torch.cos(self.pi * t))

    def step(self, t):
        return self.gamma ** (t // self.step_size)

    def custom(self, t):
        self.eps_decay = torch.tensor(4)
        if t < 0.1:
            return self.cosine(t)
        else:
            return self.exp(t) / self.cosine(0.38)

    def plot(self):
        pass

    def __repr__(self):
        repr = f"LocalScheduler(eps0={self.eps0}, lr_type={self.lr_type}"
        if self.lr_type == "exp":
            repr += f", eps_decay={self.eps_decay}"
        return repr + ")"

    def type(self):
        if self.lr_type == "exp":
            return f"exp{self.eps_decay}"
        return self.lr_type


class TorchScheduler:
    """
    Scheduler for local learning rate. Use torch.optim.lr_scheduler objects
    """

    def __init__(self, args):
        if args == []:
            args = ["1e-3", "cosine"]

        self.eps0 = float(args.pop(0))

        self.kwargs = {}

        # * Extract lr_type specific arguments
        self.lr_type = args.pop(0)
        if self.lr_type == "step":
            self.kwargs["step_size"] = int(args.pop(0))
            self.kwargs["gamma"] = float(args.pop(0))
            self.scheduler = torch.optim.lr_scheduler.StepLR

        elif self.lr_type == "exp":
            self.kwargs["gamma"] = float(args.pop(0))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR

        elif self.lr_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

        elif self.lr_type == "linear":
            self.kwargs["start_factor"] = 1
            self.kwargs["end_factor"] = 0
            self.scheduler = torch.optim.lr_scheduler.LinearLR
        else:
            raise ValueError("Unknown lr type")

        assert len(args) == 0, "Unknown arguments given to Tor"

    def __repr__(self):
        repr = f"TorchScheduler(eps0={self.eps0}, lr_type={self.lr_type}"
        for k, v in self.kwargs.items():
            repr += f", {k}={v}"
        return repr + ")"

    def __call__(self, optimizer, epochs):
        if self.lr_type == "linear":
            return self.scheduler(optimizer, total_iters=epochs, **self.kwargs)
        elif self.lr_type == "cosine":
            return self.scheduler(optimizer, T_max=epochs, **self.kwargs)
        else:  # * step and exp
            return self.scheduler(optimizer, **self.kwargs)


class AttackStrength:
    def __init__(self, args, count=1, device="cpu"):
        """
        Determines attack strength.
        Modes:
            range: ["range", <steps>, <step_size>]
                Strength is a range of <steps> linearly spaced values from 0, with step size <step_size>
            logrange: ["logrange", <steps>, <start>]
                Strength is a range of <steps> values, linearly spaced in logspace, from 0, with step size <step_size>
            custom: ["custom", <list of values>]
                Strength is a list of values given at command line
            smallest: ["smallest", <iterations>]
                Strength is dynamic to iteratively find the smallest value that still gives right answer 
        """
        self.copy = args

        if args == []:
            args = ["smallest"]
        args = DefPopList(args)

        self.mode = args.pop(0)

        if self.mode == "range":
            self.steps = int(args.pop(0, 5))
            self.step_size = float(args.pop(0, 0.2))
            self.strengths = torch.linspace(
                0, self.steps * self.step_size, self.steps+1)

        elif self.mode == "logrange":
            self.steps = int(args.pop(0, 5))
            self.start = float(args.pop(0, -4))
            self.strengths = torch.logspace(self.start, 0, self.steps)

        elif self.mode == "custom":
            self.strengths = torch.tensor([float(x) for x in args])

        elif self.mode == "smallest":
            if isinstance(device, str):
                device = torch.device(device)

            self.max_steps = int(args.pop(0, 1000))
            self.delta = float(args.pop(0, 0.01)) * torch.ones(count, device=device)
            self.min_delta = float(args.pop(0, 0.000001))
            self.strength = 0 * torch.ones(count, device=device)
            self.prev_correct = True * torch.ones(count, dtype=torch.bool, device=device)
            self.step = 0
            self.converged = False  * torch.ones(count, dtype=torch.bool, device=device)
            self.strengths = "smallest"

    def reset(self, count=1, device="cpu"):
        return AttackStrength(self.copy, count=count, device=device)

    def __repr__(self):
        return f"AttackStrength(mode={self.mode}, length={len(self.strengths)}))"

    def __iter__(self):
        for e in self.strengths:
            yield e

    def update(self, correct):
        #* has converged when delta is small enough, and model predicts wrong, but previously predicted right
        self.converged[torch.where((abs(self.delta) < self.min_delta) & correct.logical_not())] = True

        self.step += 1
        #* If all converged, stop
        if self.step > self.max_steps:
            print("Max steps reached before converged")
            self.converged += True
            return

        #* When correct flips, delta is halved and sign flipped
        self.delta = torch.where(correct != self.prev_correct, self.delta * -0.5, self.delta)
        self.prev_correct = torch.where(correct != self.prev_correct, correct, self.prev_correct)

        #* update trengths with delta
        self.strength += self.delta

        #* strength larger than 1 is converged
        self.converged[torch.where(self.strength > 1)] = True

        #* if strength is negative, go back a step, half delta and try again
        while not (self.strength >= 0).all():
            self.strength -= self.delta
            self.delta = torch.where(self.strength < 0, self.delta * 0.75, self.delta)
            self.strength += self.delta
