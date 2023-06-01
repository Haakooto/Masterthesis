import sys
sys.path.append("..")

import torch
from utils.utils import random_seed_generator as rsg
from utils.dataloader import load_data
import time
import tqdm
import numpy as np

def patchify(x, width, device):                                                                       # (imgs, height, width,  channel)
    patches_pr_dim = x.size(1) - (width - 1)
    num_patches = patches_pr_dim ** 2
    patch_pixels = width ** 2 * x.size(3)
    patches = torch.zeros((x.size(0), patch_pixels, num_patches), device=device)                      # (imgs, kernel_width * kernal_height * channels, patches)
    for row_idx in range(patches_pr_dim):
        row = x[:, row_idx: row_idx + width, :, :]                                                    # (imgs, kernel_height,  width,  channels)
        flattened = row.transpose(1, 2).reshape((x.size(0), -1))                                      # (imgs, width * kernel_height * channels)
        for col_idx in range(patches_pr_dim):
            patch = flattened[:, col_idx * x.size(3) * width: (width + col_idx) * x.size(3) * width]  # (imgs, kernel_width * kernel_height * channels)
            patches[:, :, row_idx * patches_pr_dim + col_idx] = patch

    return patches  # (imgs, kernel_width * kernal_height * channels, patches)

def minmaxnorm(x):
    """
    Scales x so that values are between 0 and 1 column-wise.
    """
    x -= x.min(dim=0)[0]  # set smallest values to 0
    maxes = x.max(dim=0)[0]  # find largest values
    maxes[(maxes < 1e-7).nonzero()] = 1  # prevent divide by zero
    return x / maxes  # scale largest values to 1

def run_patchify(x, width, batch_size, device):
    """
    Runs patchify on a batch of images.
    """
    for batch in range(0, x.size(0), batch_size):
        patches = patchify(x[batch: batch + batch_size], width, device)
        patches = patches / torch.linalg.norm(patches, dim=1, keepdims=True)

def run_unfold(x, width, batch_size, device):
    """
    Runs patchify on a batch of images.
    """
    unfold = torch.nn.Unfold(kernel_size=width, stride=1)
    for batch in range(0, x.size(0), batch_size):
        patches = unfold(x[batch: batch + batch_size])
        patches = patches / torch.linalg.norm(patches, dim=1, keepdims=True)


class Foo:
    def __init__(self):
        self.dataset = "CIFAR"
        self.seed = rsg(seed=1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(Foo(), train=True)
    imgs = 50000
    height = 32
    channels = 3
    batch_size = 25000
    width = 8
    # torch.manual_seed(rsg(seed=1))
    # x = torch.arange(imgs * height ** 2 * channels).reshape(imgs, height, height, channels).float().to(device)
    # x_rand = torch.randn(imgs, height, height, channels).to(device)
    # x_unf = x_rand.permute(0,3,1,2)  # (imgs, channels, height, width)
    
    # print(x.permute(0,3,1,2))
    # print(x.permute(0,3,1,2))
    # run_patchify(x, width, batch_size, device)
    # __import__("IPython").embed()
    # run_unfold(x_unf, width, batch_size, device)

    for batch_size in (25000, 12500, 10000, 5000, 2500, 2000, 1000, 100):
        repeats = 1
        print(dataset.x.size())
        print(f"Batch size: {batch_size}")
        print("Repeats: ", repeats)
        t1 = time.time()
        for i in range(repeats):  
            dataset.shuffle()  
            run_patchify(dataset.x, width, batch_size, device)
        t2 = time.time()
        time_patchify = t2 - t1

        t3 = time.time()
        for i in range(repeats):
            dataset.shuffle()  
            run_unfold(dataset.x_chh, width, batch_size, device)
        t4 = time.time()
        time_unfold = t4 - t3

        print(f"Avg time to patchify: {round(time_patchify / repeats, 10)}")
        print(f"Avg time to unfold  : {round(time_unfold / repeats, 10)}")
        print()

# def gpt_v1(x, pixels, num_patches, ppd, width, channels):
#     x_np = x.numpy()
#     patches_np = np.zeros((x_np.shape[0], pixels, num_patches))
#     width_channels = width * channels
#     for row_idx in range(ppd):
#         row_start = row_idx * width
#         row_end = row_start + width
#         row_slice = slice(row_start, row_end)
#         col_start = np.arange(ppd) * width_channels
#         col_end = col_start + width_channels
#         col_slices = [slice(c, c+width_channels) for c in col_start]
#         patches_np[:, :, row_idx*ppd:(row_idx+1)*ppd] = x_np[:, row_slice, :, :][:, :, :, col_slices].transpose(0, 2, 3, 1).reshape((x.shape[0], -1, ppd))
#     patches = torch.from_numpy(patches_np)
#     return patches

# def gpt_v2(x, pixels, num_patches, ppd, width, channels):
#     patches = x.unfold(1, width, width).unfold(2, width, width).reshape(x.shape[0], -1, pixels)
#     patches = patches.permute(0, 2, 1)
#     return patches

# def gpt_v3(x, pixels, num_patches, ppd, width, channels):
#     patches = torch.zeros((x.size(0), pixels, num_patches), device=x.device)
#     for row_idx in range(ppd):
#         row_start = row_idx * width
#         row_end = row_start + width
#         row_slice = slice(row_start, row_end)
#         col_start = [c * width * channels for c in range(ppd)]
#         col_end = [(c + 1) * width * channels for c in range(ppd)]
#         col_slices = [slice(cs, ce) for cs, ce in zip(col_start, col_end)]
#         patches[:, :, row_idx*ppd:(row_idx+1)*ppd] = x[:, row_slice, :, :][:, :, :, col_slices].transpose(1, 2).reshape((x.shape[0], -1, ppd))
#     return patches

# def gpt_v4(x, pixels, num_patches, ppd, width, channels):
#     patches = torch.zeros((x.size(0), pixels, num_patches), device=x.device)
#     for row_idx in range(ppd):
#         row_start = row_idx * width
#         row_end = row_start + width
#         row_slice = slice(row_start, row_end)
#         row_patches = x[:, row_slice, :, :]
#         for col_idx in range(ppd):
#             col_start = col_idx * width * channels
#             col_end = (col_idx + 1) * width * channels
#             col_slice = slice(col_start, col_end)
#             col_patches = row_patches[:, :, :, col_slice]
#             patch = col_patches.transpose(1, 2).reshape((x.size(0), -1))
#             patches[:, :, row_idx*ppd+col_idx] = patch
#     return patches

def make_indexer(pixels, num_patches, ppd, width, channels, device):
    x = torch.arange(32 * 32 * 3, dtype=int).reshape(1, 32, 32, 3)
    Idx = patchify(x, pixels, num_patches, ppd, width, channels, device, dtype=int)
    return Idx[0]

def indexify_num(x, Idx, pixels, num_patches, device):
    patches = torch.zeros((x.size(0), pixels, num_patches), device=device)
    x = x.flatten(1)
    for i in range(num_patches):
        patches[:, :, i] = x[:, Idx[:, i]]
    return patches

def indexify_pix(x, Idx, pixels, num_patches, device):
    # patches = torch.zeros((x.size(0), pixels * num_patches), device=device)
    x = x.flatten(1)
    # print(Idx.shape)
    patches = x[:, Idx]
    # for i in range(pixels):
    #     patches[:, i] = x[:, Idx[i]]
    # print(x.device)
    # print(Idx.device)
    # print(patches.device)
    return patches.reshape((x.size(0), pixels, num_patches))

def patchify(x, pixels, num_patches, ppd, width, channels, device, dtype=None):
    patches = torch.zeros((x.size(0), pixels, num_patches), dtype=x.dtype if dtype is None else dtype, device=device)                              
    for row_idx in range(ppd):
        row = x[:, row_idx: row_idx+width, :, :]                                                                        
        flattened = row.transpose(1, 2).reshape((x.size(0), -1))                                                             
        for col_idx in range(ppd):
            patch = flattened[:, col_idx * channels * width: (width + col_idx) * channels * width]  
            patches[:, :, row_idx * ppd + col_idx] = patch
    return patches

def main2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(Foo(), train=True)

    x = dataset.x
    imgs = 50000
    height = 32
    channels = 3
    batch_size = 1000

    for width in [4, 8, 12]:
        print(f"width: {width}")
        patches_pr_dim = x.size(1) - (width - 1)
        num_patches = patches_pr_dim ** 2
        patch_pixels = width ** 2 * x.size(3)

        Idx = make_indexer(patch_pixels, num_patches, patches_pr_dim, width, channels, device).flatten()

        for bidx in range(0, imgs, batch_size):
            batch = x[bidx: bidx+batch_size].to(device)
            patches1 = patchify(batch, patch_pixels, num_patches, patches_pr_dim, width, channels, device)
            # patches2 = indexify_num(batch, Idx, patch_pixels, num_patches, device)
            patches3 = indexify_pix(batch, Idx, patch_pixels, num_patches, device)
            # assert torch.allclose(patches1, patches2)
            assert torch.allclose(patches1, patches3)
            # assert torch.allclose(patches2, patches3)
        print("they are the same")

        time_patch = time.time()
        for bidx in range(0, imgs, batch_size):
            batch = x[bidx: bidx+batch_size]
            patches_old = patchify(batch, patch_pixels, num_patches, patches_pr_dim, width, channels, device)
        time_patch = time.time() - time_patch

        # time_idx_num = time.time()
        # for bidx in range(0, imgs, batch_size):
        #     batch = x[bidx: bidx+batch_size]
        #     patches_pat = indexify_num(batch, Idx, patch_pixels, num_patches, device)
        # time_idx_num = time.time() - time_idx_num
        
        time_idx_pix = time.time()
        for bidx in range(0, imgs, batch_size):
            batch = x[bidx: bidx+batch_size]
            patches_pix = indexify_pix(batch, Idx, patch_pixels, num_patches, device)
        time_idx_pix = time.time() - time_idx_pix

        print(f"patchify: {time_patch}, indexify_pix: {time_idx_pix}")
        print(f"patch/pix: {time_patch/time_idx_pix}")
        


    # patches = indexify(mini_x, Idx, patch_pixels, num_patches)
    # mine = mypatchify(mini_x, patch_pixels, num_patches, patches_pr_dim, width, channels)


if __name__ == "__main__":  
    main2()
    