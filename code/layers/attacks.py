"""
This file implements different attack-methods
Should not be run directly, rather used by model_attack.py
"""
import torch
from torch.nn import functional as FF
from tqdm import tqdm
import numpy as np

class fgsm_Manager:
    def __init__(self, args, x, y, store_examples=False):
        self.x = x
        self.y = y
        self.args = args
        self.model = args.model

        count = len(self.x)
        device = self.args.device

        self.max_steps = 1000
        self.delta = 0.01 * torch.ones(count, device=device)
        self.min_delta = 1e-6
        self.strength = 0 * torch.ones(count, device=device)
        self.prev_correct = True * torch.ones(count, dtype=torch.bool, device=device)
        self.step = 0
        self.converged = False * torch.ones(count, dtype=torch.bool, device=device)
            
        if store_examples:
            self.storage = {"imgs": np.zeros((self.max_steps, *self.x.shape)),
                            "preds": np.zeros((self.max_steps, count, 10)),
                            "strength": np.zeros((self.max_steps, count)),
                            "convd": np.zeros((self.max_steps, count))
            ,}
            self.storage["imgs"][0] = self.x.cpu().numpy()
            self.storage["preds"][0] = self.model(self.x).exp().detach().cpu().numpy()
            self.storage["strength"][0] = self.strength.cpu().numpy()
            self.storage["convd"][0] = self.converged.cpu().numpy()

    def attack(self):
        self.x.requires_grad = True
        output = self.model(self.x)
        loss = FF.nll_loss(output, self.y, reduction='sum')
        loss.backward()
        dx = self.x.grad.detach().sign()

        i = 0
        pbar = tqdm(total=len(self.x), desc="Step: 0", leave=False)
        while not self.converged.all():
            i += 1
            if self.args.BP:
                Px = self.x + dx * self.strength.reshape(-1, 1, 1, 1)
            else:
                Px = self.x + dx * self.strength.reshape(-1, 1)
            clamped = Px.clamp(min=0, max=1)

            with torch.no_grad():
                pred = self.model(clamped).argmax(dim=1)
                correct = pred.eq(self.y)
                if not self.update(correct):
                    return False

            pbar.reset()
            pbar.update(self.converged.sum().item())
            pbar.set_description(f"Step: {i}")

        pbar.close()
        return True

    def attack_final(self):
        self.x.requires_grad = True
        output = self.model(self.x)
        loss = FF.nll_loss(output, self.y, reduction='sum')
        loss.backward()
        dx = self.x.grad.detach().sign()

        i = 0
        while not self.converged.all():
            i += 1
            if self.args.BP:
                Px = self.x + dx * self.strength.reshape(-1, 1, 1, 1)
            else:
                Px = self.x + dx * self.strength.reshape(-1, 1)
            clamped = Px.clamp(min=0, max=1)

            self.storage["imgs"][i] = clamped.detach().cpu().numpy()

            with torch.no_grad():
                pred = self.model(clamped)
                correct = pred.argmax(dim=1).eq(self.y)
                if not self.update(correct):
                    return False
                
                self.storage["preds"][i] = pred.exp().detach().cpu().numpy()
                self.storage["strength"][i] = self.strength.detach().cpu().numpy()
                self.storage["convd"][i] = self.converged.detach().cpu().numpy()
        print(i)
        self.storage["imgs"] = self.storage["imgs"][:i+1]
        self.storage["preds"] = self.storage["preds"][:i+1]
        self.storage["strength"] = self.storage["strength"][:i+1]
        self.storage["convd"] = self.storage["convd"][:i+1]

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
        i = 0

        self.strength -= self.delta
        self.delta = torch.where((self.strength + self.delta) <= 0, -self.delta * 0.75, self.delta)
        self.strength += self.delta
    
        return True

    def save_metrics(self):
        for b in range(len(self.x)):
            if (rho:=self.strength[b]) > 1:
                continue
            self.args.results.rhos["all"].append(rho.item())
            self.args.results.rhos[self.y[b].item()].append(rho.item())

class pgd_Manager:
    def __init__(self, args, x, y):
        self.x0 = x
        self.x = x.clone()
        self.y = y
        self.args = args
        self.model = args.model

        self.step_size = args.step
        self.proj_size = args.size
        self.max_steps = 700
        self.storage = 100
        self.tol = 1e-6

        self.conv_by_mis = 0
        self.conv_by_tol = 0

        self.prev_correct = True * torch.ones(len(self.x), dtype=torch.bool, device=self.args.device)
        self.converged = False * torch.ones(len(self.x), dtype=torch.bool, device=self.args.device)
        self.steps = torch.zeros(len(self.x), dtype=int, device=self.args.device)
        self.correct = torch.zeros(len(self.x), dtype=torch.bool, device=self.args.device)
        self.history = -torch.ones((self.storage, *x.shape), device=self.args.device)

    def attack(self):
        self.history[0] = self.x0
        i = 0
        pbar = tqdm(total=len(self.x), desc="Step: 0", leave=False)
        while not self.converged.all() and i < self.max_steps-1:
            i += 1
            self.x.requires_grad = True

            out = self.model(self.x[~self.converged])
            loss = FF.nll_loss(out, self.y[~self.converged], reduction='sum')
            loss.backward()
            dx = self.x.grad.detach().sign()

            assert dx[self.converged].sum() == 0

            self.x.requires_grad = False
            #* update the images where they have not converged
            self.x = self.x + dx * self.step_size
            #* project the images to the L_infinity-ball around the original image
            self.x = self.x0 + torch.clamp(self.x - self.x0, min=-self.proj_size, max=self.proj_size)
            #* clamp the images to be between 0 and 1
            self.x = self.x.clamp(min=0, max=1)

            self.correct *= False  #* Set all to false
            with torch.no_grad():
                #* eval model where not converged
                pred = self.model(self.x[~self.converged]).argmax(dim=1)
                #* set correct to true where prediction is correct, where not converged
                self.correct[~self.converged] = pred.eq(self.y[~self.converged])
                self.update()
            pbar.reset()
            pbar.update(self.converged.sum().item())
            pbar.set_description(f"Step: {i}")

        pbar.close()

        self.prev_correct.cpu()
        self.converged.cpu()
        self.correct.cpu()
        self.history.cpu()

        #! Need to only calculate this for misclassified images
        hits = self.model(self.x).argmax(dim=1).eq(self.y)
        self.rhos = torch.linalg.norm((self.x - self.x0).flatten(1), ord=torch.inf, dim=1)
        self.rhos[hits] = 1
        self.steps[hits] = -1

        return True


    def update(self):
        #* Save position
        #* Has converged when misclassification happens
        self.steps[~self.converged] += 1  #* increase steps for those that have not converged
        self.conv_by_mis += (~self.correct[~self.converged]).sum().item()
        self.converged[~self.correct] = True

        #* If image revisits the same place, it will loop forever, so we stop it
        #* Calculate the inf-norm between the current image and all previous images
        #* First extract the prev history, subtract the images where not converged,
        #* and flatten the images because norm can only be calculated over 1, 2 or all axis, not 3???
        dists = torch.linalg.norm((self.history[:, ~self.converged] - self.x[~self.converged]).flatten(2), dim=2, ord=torch.inf)

        self.converged[~self.converged] = dists.min(dim=0).values < self.tol
        self.conv_by_tol += (dists.min(dim=0).values < self.tol).sum().item()

        self.history[(self.steps-1) % self.storage] = self.x

    def save_metrics(self):
        for rho, y, steps in zip(self.rhos, self.y, self.steps):
            self.args.results.rhos["all"].append(rho.item())
            self.args.results.rhos[y.item()].append(rho.item())
            self.args.results.steps["all"].append(steps)
            self.args.results.steps[y.item()].append(steps)
