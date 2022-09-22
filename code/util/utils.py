'''Some helper functions for PyTorch, including:
    - stft: tranform 1d signal to time-frequency image using short-time fourier transform
    - preprocessing:  perform stft for each epoch in a batch of sequential epochs
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from random import shuffle

def lin_tri_filter_shape(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Compute a linear-filterbank. The filters are stored in the rows, the columns correspond to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    #lowmel = self.hz2mel(lowfreq)
    #highmel = self.hz2mel(highfreq)
    #melpoints = np.linspace(lowmel,highmel,nfilt+2)
    hzpoints = np.linspace(lowfreq,highfreq,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*hzpoints/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    fbank = np.transpose(fbank)
    fbank.astype(np.float32)
    return fbank

def stft(signal, sample_rate=100, frame_size=2, frame_stride=1, winfunc=np.hamming, NFFT=256):
    '''
    short-time fourier transform (STFT)
    represent 1d signal to time-frequency image representation
    In SeqSleepNet for each x: [30*100] -> [29, 129] --- 29=1+(3000-100*2)/100*1;  129=256/2+1
    '''
    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

    return pow_frames

def preprocessing(x):
    # x [bs, seq_len, ch, 30*100]
    # return [bs, seq_len, ch, 29, 129]
    bslist = []
    for b in range(x.shape[0]):
        seqlist = []
        for s in range(x.shape[1]):
            chlist = []
            for c in range(x.shape[2]):
                tmp = stft(x[b,s,c,:])
                tmp = torch.from_numpy(tmp)
                tmp = torch.unsqueeze(tmp, 0)
                chlist.append(tmp)
            chout = torch.cat(chlist, 0) # [ch, 29, 129]
            chout = torch.unsqueeze(chout, 0) # [1, ch, 29, 129]
            seqlist.append(chout)
        seqout = torch.cat(seqlist, 0)
        seqout = seqout.unsqueeze(0)
        bslist.append(seqout)
    out = torch.cat(bslist, 0) # out [bs, seq_len, 29, 129]
    out = out.type(torch.float)
    return out

def prepare_stft_loader(loader_dir):
    loaders = []
    for root, dirs, files in os.walk(loader_dir):
        if files:
            for file_ in files:
                loaders.append(os.path.join(loader_dir, root, str(file_)))

    for loader in loaders:
        slash   = loader.split('/')
        subdir  = '/'.join(str(x) for x in slash[:-1])+'/stft'
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        df      = torch.load(loader)
        x, y    = df.dataset.tensors[0], df.dataset.tensors[1]
        idx     = list(range(0, 6000, 2))
        x       = x[:, :, idx]
        x       = preprocessing(x)
        dataset = TensorDataset(x, y)
        loader_ = DataLoader(dataset, batch_size=df.batch_size)

        torch.save(loader_, subdir+'/'+slash[-1])

def combine_loader(loader_list):
    xlist, ylist = [], []
    for loader in loader_list:
        x, y = loader.dataset.tensors[0], loader.dataset.tensors[1]
        xlist.append(x), ylist.append(y)
    x, y    = torch.cat(xlist, 0), torch.cat(ylist, 0)
    dataset = TensorDataset(x, y)
    loader  = DataLoader(dataset, batch_size=loader_list[0].batch_size)

    return loader

def make_sleep_edf_dataloader(dataset_dir, batch_size):
    files = os.listdir(dataset_dir)
    x, y = [], []
    for f in files:
        df = np.load(os.path.join(dataset_dir, f))
        data = df['x']
        data = data.transpose(0,2,1)
        label = df['y']
        x.append(data)
        y.append(label)

    x, y = tuple(x), tuple(y)
    x, y = np.concatenate(x), np.concatenate(y)
    torch_x, torch_y = torch.from_numpy(x), torch.from_numpy(y)
    dataset = TensorDataset(torch_x, torch_y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def seq_cel(pred, gt, class_num):
    # seq cross entropy loss
    criterion= nn.CrossEntropyLoss()
    loss  = criterion(pred, gt)
    total = torch.numel(gt)
    corr  = torch.sum(torch.eq(pred.max(1)[1], gt)).item()
    return loss, corr, total

def gdl(pred, gt, class_num):
    # generalized dice loss
    # pred: bs, class_num, seq_len
    # gt  : bs, seq_len
    onehot_y = F.one_hot(gt.long(), class_num)
    pred_t   = pred.permute(0, 2, 1)

    intersection = torch.sum(onehot_y * pred_t)
    union        = torch.sum(onehot_y + pred_t)
    loss         = 1 - 2 * intersection / (union * class_num)

    pred  = torch.argmax(pred, dim=1)
    corr  = torch.sum(torch.eq(pred.long(), gt.long())).item()
    total = torch.numel(gt)

    return loss, corr, total

def make_seq_stft_loader(loader, seq_len, stride):
    # input : loader of size [#n, 1, 29, 129], [#n]
    # return: loader of size [#n, seq_len, 29, 129], [#n]

    x, y   = loader.dataset.tensors[0], loader.dataset.tensors[1]
    dim    = x.shape[-1]
    idx    = gen_seq(x.shape[0], seq_len, stride)
    xx, yy = [x[i:i+seq_len, :, :, :] for i in idx], [y[i:i+seq_len] for i in idx]
    xx     = [x.reshape(1, seq_len, x.shape[2], x.shape[3]) for x in xx]
    yy     = [y.unsqueeze(0) for y in yy]

    xx, yy = torch.cat(xx), torch.cat(yy)
    dataset= TensorDataset(xx, yy)
    loader = DataLoader(dataset, batch_size=loader.batch_size)
    return loader

def make_seq_loader(loader, seq_len, stride):
    # input : loader of size [#n, 1, #dim], [#n]
    # return: loader of size [#n, seq_len, #dim], [#n]

    x, y   = loader.dataset.tensors[0], loader.dataset.tensors[1]
    dim    = x.shape[-1]
    idx    = gen_seq(x.shape[0], seq_len, stride)
    xx, yy = [x[i:i+seq_len, :, :] for i in idx], [y[i:i+seq_len] for i in idx]
    xx     = [x.reshape(-1, x.shape[0]*x.shape[2]) for x in xx]
    xx, yy = [x.unsqueeze(0) for x in xx], [y.unsqueeze(0) for y in yy]
    xx, yy = torch.cat(xx), torch.cat(yy)
    xx     = torch.reshape(xx, [-1, seq_len, dim]) # [#n, seq_len, dim]
    dataset= TensorDataset(xx, yy)
    loader = DataLoader(dataset, batch_size=loader.batch_size)
    return loader

def gen_seq(n, seq_len, stride):
    res = []
    for i in range(0, n, stride):
        if i + seq_len <= n:
            res.append(i)
    return res

def make_dataloader(dataset_dir, files, channel, batch_size, shuffle):
    x, y = [], []
    for f in files:
        matf = os.path.join(dataset_dir, f)
        mat  = loadmat(matf)
        data = mat['data'][:,:,channel]
        data = data.reshape(data.shape[0], 1, data.shape[1])
        #print(data.shape)
        label = mat['labels']
        x.append(data)
        y.append(label)
    x, y = tuple(x), tuple(y)
    x, y = np.vstack(x), np.vstack(y)
    torch_x, torch_y = torch.from_numpy(x), torch.from_numpy(y)
    dataset = TensorDataset(torch_x, torch_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
import math
from glob import glob


def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"util/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data,r_permute

def load_folds_data(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print(files)
    if "78" in np_data_path:
        r_p_path = r"util/r_permute_78.npy"
    elif "20" in np_data_path:
        r_p_path = r"util/r_permute_20.npy"
    else:
        r_p_path = r"util/r_permute_isruc.npy"
    
    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        print(r_permute)
        # r_permute = np.array(range(20))
        # print(r_permute.shape)
    else:
        print ("============== ERROR =================")


    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype=object)
    print("files_pairs",files_pairs.shape)
    files_pairs = files_pairs[r_permute]
    # print("files_pairs",files_pairs)

    train_files = np.array_split(files_pairs, n_folds)
    print("train_files",train_files[0])

    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        # print("files_pairs2",files_pairs2)
        # print("----------------------------------------------------------------------")
        # print("subject_files",subject_files)
        # print(set(files_pairs2) - set(subject_files))
        # training_files = [i for i in files_pairs2 if i not in subject_files]
        training_files = list(set(files_pairs2) - set(subject_files))
        # print("training_files:",training_files)
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data,r_permute

def load_folds_data_isruc(np_data_path, n_folds):
    
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"util/r_permute_isruc.npy"
    
    if os.path.exists(r_p_path):
        # r_permute = np.load(r_p_path)
        r_permute = np.array(range(10))
        print('r_permute',r_permute)
        # r_permute = np.array(range(20))
        # print(r_permute.shape)
    else:
        print ("============== ERROR =================")
    
   

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        # print('file_name',file_name)
        file_num = file_name[0:-3]
        # print('file_num',file_num)
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype=object)
    print("files_pairs",files_pairs)
    files_pairs = files_pairs[r_permute]
    # print("files_pairs",files_pairs)

    train_files = np.array_split(files_pairs, n_folds)
    print("train_files",train_files[0])

    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        # print("files_pairs2",files_pairs2)
        # print("----------------------------------------------------------------------")
        # print("subject_files",subject_files)
        # print(set(files_pairs2) - set(subject_files))
        # training_files = [i for i in files_pairs2 if i not in subject_files]
        training_files = list(set(files_pairs2) - set(subject_files))
        # print("training_files:",training_files)
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data,r_permute

def load_folds_data_isruc100(np_data_path, n_folds):
    
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"util/r_permute_isruc.npy"
    
    if os.path.exists(r_p_path):
        # r_permute = np.load(r_p_path)
        r_permute = np.array(range(100))
        # print('r_permute',r_permute)
        # r_permute = np.array(range(20))
        # print(r_permute.shape)
    else:
        print ("============== ERROR =================")
    file_random = [[41, 85, 38, 22, 75, 77, 43, 84, 32, 14, 58, 80, 97, 46, 25, 13, 44,
       79, 12, 81], [64, 71, 55, 72, 98, 69, 17, 49, 76, 29, 60, 51, 92, 89, 10, 24, 39,
        3,  9, 78], [86, 28, 95, 21,  7, 16, 73, 23, 96,  4, 67, 68, 52, 87,  8, 27, 74,
       99, 61, 91], [66, 56, 93, 70, 33, 47, 15,  1, 57, 53, 94, 37, 65, 90, 26,  6, 35,
       20, 34, 48], [83, 63, 59, 31, 62,  2, 30, 50, 40, 54, 82, 18, 42, 88, 36, 19,  5,
       11, 45,  0]]
   

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        # print('file_name',file_name)
        file_num = file_name[0:-3]
        # print('file_num',file_num)
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs,dtype=object)
    print("files_pairs",files_pairs)
    files_pairs = files_pairs[r_permute]
    # print("files_pairs",files_pairs)

    # train_files = np.array_split(files_pairs, n_folds)
    # print("train_files",train_files[0])

    folds_data = {}
    for fold_id in range(n_folds):
        subject = file_random[fold_id]
        training_files = [files_pairs[i] for i in subject]
        
        
        # print("files_pairs2",files_pairs2)
        # print("----------------------------------------------------------------------")
        # print("subject_files",subject_files)
        # print(set(files_pairs2) - set(subject_files))
        # training_files = [i for i in files_pairs2 if i not in subject_files]
        # subject_files = [i for i in files_pairs if i not in training_files]
        training_files = [item for sublist in training_files for item in sublist]
        # print('training_files',training_files)
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        subject_files = list(set(files_pairs2) - set(training_files))
        # print("subject_files:",subject_files)
        # training_files = training_files.tolist()
        # subject_files = subject_files.tolist()
        folds_data[fold_id] = [training_files, subject_files]
    
    return folds_data,r_permute


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
