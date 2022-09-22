import numpy as np
import torch
from scipy import signal
import random
from random import choice

def DataTransform(sample,augment_type):#, config):

    # weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    if augment_type=='weak':
        aug = scaling(sample)
    elif augment_type=='strong':
        aug = jitter(permutation(sample))
    return aug
    # return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

# def ramdom_augmentor(sample, config):
#     augmentations=[time_shift,masking,dc_shift,amplitude_scale,bandstop_filter,additive_noise]
#     # aug_number = random.sample(range(1,6), 2)
#     aug1=choice(augmentations)(sample)
#     aug2=choice(augmentations)(sample)

#     return aug1,aug2

# def time_shift(x,shift_length):
#     x=np.roll(x, shift_length, axis=0)
#     return x

# def masking(x,mask_length,first_point):
#     mask=np.zeros(mask_length)
#     x[first_point:first_point+mask_length]=mask
#     return x

# def dc_shift(x,shift_length):
#     x=np.roll(x, shift_length, axis=1)
#     return x

# def amplitude_scale(x,scale):  
#     # x=x*scale
#     # return x
#     factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
#     ai = []
#     for i in range(x.shape[1]):
#         xi = x[:, i, :]
#         ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
#     return np.concatenate((ai), axis=1)

# def bandstop_filter(x, a, b):  #5 Hz width
#     x = signal.filtfilt(b, a, x)  #x为要过滤的信号
#     return x

# def additive_noise(x,mu=0,std=0.1):
#     noise = np.random.normal(mu, std, size = x.shape)
#     x_noisy = x + noise
#     return x_noisy 