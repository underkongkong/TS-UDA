import torch
from torch.utils.data import Dataset
import os
import numpy as np
from dataset.data_psd import PSD
# from data_psd import PSD
from dataset.data_representation import EEG_Spectral_spatial_representation
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
# from dataset.sobi import sobi
import copy
from .augmentations import DataTransform

class LoadDataset_from_numpy_augment(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,augment_type=None):
        super(LoadDataset_from_numpy_augment, self).__init__()
        self.augment_type = augment_type
        # self.SS=SS
        X_train = np.load(np_dataset[0])["x"]
        # print(X_train.shape)

        y_train = np.load(np_dataset[0])["y"]
        # y_train = np.argmax(np.load(np_dataset[0])["y"],axis=1)#isruc
        # print(y_train.shape)
        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            # if self.SS==8 or self.SS==9 or self.SS == 10 or self.SS == 11 :
            #     y_train = np.append(y_train, np.argmax(np.load(np_file)["y"],axis=1))
            
            y_train = np.append(y_train, np.load(np_file)["y"])
            # print(y_train.shape)
        # if len(X_train.shape) == 3 :
        #     if X_train.shape[1] != 1:
        #         # X_train = X_train.permute(0, 2, 1)
        #         X_train = np.transpose(X_train,(0, 2, 1))
        #         # print('self.x_data.shape1',self.x_data.shape)
        # else:
        #     X_train = X_train.unsqueeze(1)
        if "isruc" in np_dataset[0]:
            X_train = X_train[:,[1,4],:]#[:,np.newaxis,:]
        elif "sleepedf" in np_dataset[0]:
            X_train = X_train.transpose(0,2,1)[:,0:2,:]#[:,np.newaxis,:]
        print('X_train.shape',X_train.shape)
        
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        self.len = self.x_data.shape[0]
        # if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
        # self.aug1, self.aug2 = DataTransform(self.x_data)
        if self.augment_type =='weak':
            self.aug1 = DataTransform(self.x_data,self.augment_type)
        elif self.augment_type =='strong':
            self.aug2 = DataTransform(self.x_data,self.augment_type)

    def __getitem__(self, index):
        # if "weak" in np_dataset:
        # return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        if self.augment_type =='weak':
            return  self.aug1[index],self.y_data[index]
            # return  self.x_data[index],self.y_data[index]
        elif self.augment_type =='strong':
            return  self.aug2[index],self.y_data[index]
            # return  self.x_data[index],self.y_data[index]
        elif self.augment_type == 'None':
            return  self.x_data[index],self.y_data[index]
        # return  self.aug1[index], self.aug2[index],self.y_data[index]
    def __len__(self):
        return self.len

def data_generator_augment(source_files,target_files,batch_size,workers=2):
    train_sub = 0.8
    # test_sub = 1-train_sub

    source_weak_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='weak')
    source_strong_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='strong')
    source_test_dataset = LoadDataset_from_numpy_augment(source_files[int(train_sub*len(source_files)):],augment_type='None')


    target_weak_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(source_files))],augment_type='weak')
    target_strong_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(source_files))],augment_type='strong')
    target_test_dataset = LoadDataset_from_numpy_augment(target_files[int(train_sub*len(source_files)):],augment_type='None')

    all_ys = np.concatenate((source_weak_dataset.y_data, target_strong_dataset.y_data))
    all_ys = all_ys.tolist()

    num_classes = len(np.unique(all_ys))
    # print('num_classes',np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    source_weak_loader = torch.utils.data.DataLoader(dataset=source_weak_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=workers)
    
    source_strong_loader = torch.utils.data.DataLoader(dataset=source_strong_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=workers)

    source_test_loader = torch.utils.data.DataLoader(dataset=source_test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=workers)                                           

    target_weak_loader = torch.utils.data.DataLoader(dataset=target_weak_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=workers)

    target_strong_loader = torch.utils.data.DataLoader(dataset=target_strong_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=workers)
                                        
    target_test_loader = torch.utils.data.DataLoader(dataset=target_test_dataset,
                                              batch_size=batch_size,
                                              shuffle = False,
                                              drop_last=False,
                                              num_workers=workers)
    return (source_weak_loader, source_strong_loader,source_test_loader),(target_weak_loader,target_strong_loader, target_test_loader)
    # return source_loader,target_loader,counts





