from collections import namedtuple
import random

import torch
import torchaudio
from torchaudio import transforms
from data.SparseImageWarp import sparse_image_warp
def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device
    
    y = num_rows//2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len
    
    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)

def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    batch = cloned.shape[0]
    for i in range(0, num_masks):       
        for j in range(batch): 
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (replace_with_zero): cloned[j][f_zero:mask_end] = 0
            else: cloned[j][f_zero:mask_end] = cloned[j].mean()
    
    return cloned
    
def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    batch = cloned.shape[0]
    for i in range(0, num_masks):
        for j in range(batch):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero): cloned[j][:,t_zero:mask_end] = 0
            else: cloned[j][:,t_zero:mask_end] = cloned[j].mean()
    return cloned
def spec_augmentation(x,F, T, num_masks = 2):
    x = x.permute(0,2,1)
    combined = time_mask(freq_mask(x, F, num_masks=num_masks), T, num_masks=num_masks)
    combined = combined.permute(0,2,1)
    return combined
