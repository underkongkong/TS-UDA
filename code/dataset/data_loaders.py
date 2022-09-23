import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform

class LoadDataset_from_numpy_augment(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,augment_type=None):
        super(LoadDataset_from_numpy_augment, self).__init__()
        self.augment_type = augment_type

        X_train = np.load(np_dataset[0])["x"]
        # print(X_train.shape)
        y_train = np.load(np_dataset[0])["y"]
        # y_train = np.argmax(np.load(np_dataset[0])["y"],axis=1)#isruc
        # print(X_train.shape)
        # print(y_train.shape)
        for np_file in np_dataset[1:]:
            # print(np_file)
            temp_data=np.load(np_file)["x"]
            if temp_data.shape[2] == X_train.shape[2]:
                X_train = np.vstack((X_train, temp_data))
            # print(np.load(np_file)["y"].shape)
            if "isruc" in np_dataset[0]:
                y_train = np.vstack((y_train, np.load(np_file)["y"]))
            elif "sleepedf" or 'shhs' in np_dataset[0]:
                y_train = np.hstack((y_train, np.load(np_file)["y"]))
        # print(y_train.shape)

        # isruc : ["F3_A2", "C3_A2", "F4_A1", "C4_A1", "O1_A2", "O2_A1", "ROC_A1", "X1"]
        # sleepedf : [fpz-cz, pz-oz, eog, emg]
        # shhs : [c3, c4, eog, emg]
        if "isruc" in np_dataset[0]:
            X_train = X_train[:,1,:][:,np.newaxis,:]
            y_train = np.argmax(y_train,axis=1)

        elif "sleepedf" in np_dataset[0]:
            X_train = X_train.transpose(0,2,1)[:,0,:][:,np.newaxis,:]

        elif "shhs" in np_dataset[0]:
            X_train = X_train.transpose(0,2,1)[:,0,:][:,np.newaxis,:]
    
        # print('X_train.shape',X_train.shape)
        
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # print(self.x_data.shape)
        self.len = self.x_data.shape[0]
        # if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
        # self.aug1, self.aug2 = DataTransform(self.x_data)
        if self.augment_type =='weak' or self.augment_type =='strong':
            self.x_data = DataTransform(self.x_data,self.augment_type)

    def __getitem__(self, index):
        # if "weak" in np_dataset:
        # return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        return  self.x_data[index],self.y_data[index]
        # return  self.aug1[index], self.aug2[index],self.y_data[index]

    def __len__(self):
        return self.len
        
def dataset_generator_augment(source_files,target_files):
    train_sub = 0.8
    # test_sub = 1-train_sub

    source_weak_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='weak')
    source_strong_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='strong')
    source_test_dataset = LoadDataset_from_numpy_augment(source_files[int(train_sub*len(source_files)):],augment_type='None')
    print('source train data:',source_weak_dataset.x_data.shape)
    print('source test data:',source_test_dataset.x_data.shape)
    print('target y train shape:',source_weak_dataset.y_data.shape)
    print('target y test shape:',source_test_dataset.y_data.shape)

    target_weak_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(target_files))],augment_type='weak')
    target_strong_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(target_files))],augment_type='strong')
    target_test_dataset = LoadDataset_from_numpy_augment(target_files[int(train_sub*len(target_files)):],augment_type='None')
    print('target train data:',target_weak_dataset.x_data.shape)
    print('target test data:',target_test_dataset.x_data.shape)
    print('target y train shape:',target_weak_dataset.y_data.shape)
    print('target y test shape:',target_test_dataset.y_data.shape)
    return source_weak_dataset,source_strong_dataset,source_test_dataset,target_weak_dataset,target_strong_dataset,target_test_dataset

def dataset_generator(source_files,target_files):
    source_dataset = LoadDataset_from_numpy_augment(source_files,augment_type='None')
    target_dataset = LoadDataset_from_numpy_augment(target_files,augment_type='None')
    return source_dataset,target_dataset

def data_generator_augment(source_files,target_files,batch_size,workers=2):
    train_sub = 0.8
    # test_sub = 1-train_sub

    source_weak_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='weak')
    source_strong_dataset = LoadDataset_from_numpy_augment(source_files[:int(train_sub*len(source_files))],augment_type='strong')
    source_test_dataset = LoadDataset_from_numpy_augment(source_files[int(train_sub*len(source_files)):],augment_type='None')
    print('source train data:',source_weak_dataset.x_data.shape)
    print('source test data:',source_test_dataset.x_data.shape)
    print('target y train shape:',source_weak_dataset.y_data.shape)
    print('target y test shape:',source_test_dataset.y_data.shape)

    target_weak_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(target_files))],augment_type='weak')
    target_strong_dataset = LoadDataset_from_numpy_augment(target_files[:int(train_sub*len(target_files))],augment_type='strong')
    target_test_dataset = LoadDataset_from_numpy_augment(target_files[int(train_sub*len(target_files)):],augment_type='None')
    print('target train data:',target_weak_dataset.x_data.shape)
    print('target test data:',target_test_dataset.x_data.shape)
    print('target y train shape:',target_weak_dataset.y_data.shape)
    print('target y test shape:',target_test_dataset.y_data.shape)

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





