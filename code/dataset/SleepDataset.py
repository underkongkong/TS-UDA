
from torch.utils.data import Dataset, dataloader
import numpy as np
import mne
from xml.dom.minidom import parse
from dataset.data_representation import EEG_Spectral_spatial_representation
import copy
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA, FastICA

class SleepDataset(Dataset):

    def __init__(self,
                task:str , # 'children' or 'adults'
                data_path: str = None,
                target_channels= ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1','Chin 1-Chin 2'],
                subject_list = [0,1,2], # 0~2, 3 subjects
                downsample: int = None,
                scaler = None,
                ):
        print('target_channels',target_channels)
        assert task=='children' or task=='adults'
        self.task = task

        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = f'/home/brain/code/SleepStagingPaper/data_org/{self.task}/'
        
        self.target_channels = target_channels
        self.subject_list = subject_list
        self.time_length = 30 # 30s
    
        self.downsample = downsample

        # event_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 5}

        self.X0, self.X1,self.X2, self.y = self.get_X_y(subject_list=self.subject_list)
        print(np.unique(self.y, return_counts=True))

        if scaler is not None:
            for i in range(self.X.shape[0]):
                self.X[i] = scaler(self.X[i].T).T
        
       
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X0[idx], self.X1[idx],self.X2[idx],self.y[idx]



    def fastICA(data):
        tmin, tmax = -0.1, 0.3
        ica_data= copy.deepcopy(data[i,:,:])
        for i in range(data.shape[0]):
            X = data[i,:,:]
            ica = UnsupervisedSpatialFilter(
            FastICA(30, whiten='unit-variance'), average=False)
            ica_data[i,:,:] = ica.fit_transform(X)
            # ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
            #                   mne.create_info(30, sfreq=100,
            #                                   ch_types='eeg'), tmin=tmin)
        return ica_data

    def get_X_y(self, subject_list):
        X_data = []
        y_data = []
        # ss_data = []
        for subject in subject_list:
            subject_label = self.get_label(self.data_path, subject)
            subject_data = self.get_data(self.data_path, subject)
            # subject_ss=EEG_Spectral_spatial_representation(subject_data[:,2:8,:])
            # print(subject_data.shape, subject_label.shape)
            '''
            (1023, 1, 7680) (1023,)
            (1118, 1, 7680) (1118,)
            (1030, 1, 7680) (1031,)
            '''
            if subject_data.shape[0] != subject_label.shape[0]:
                subject_label = subject_label[:-1] # delete the last label
                assert subject_data.shape[0] == subject_label.shape[0] # check
            
            X_data.append(subject_data)
            y_data.append(subject_label)
            # ss_data.append(subject_ss)
        # print('X_data.shape',X_data.shape)   
        X_data = np.concatenate(X_data)
        ICA_data= self.fastICA(X_data)

        self.y_data = np.concatenate(y_data)
        # ss_data = np.concatenate(ss_data)
        print('X_data.shape',X_data.shape)
        self.X = [[],[],[]]
        # self.X[3] = ss_data
        self.X[2] = X_data[:,8,:]#EMG
        # self.X[3] = torch.from_numpy(self.X[3])
        self.X[0] = X_data[:,2:8,:]#EEG
        self.X[1] = X_data[:,0:2,:]#EOG
        
        return self.X[0],self.X[1],self.X[2],self.y_data

    def get_label(self,data_path, subject):
        if self.task=='adults':
            filename = f"{data_path}0{subject+1}.XML"
        elif self.task=='children':
            filename = f"{data_path}0{subject+1}.xml" 
        
        label = []
        DOMTree = parse(filename)
        root = DOMTree.documentElement
        sleep_stages = root.getElementsByTagName('SleepStage')
        # print('number of sleep stages: ', len(sleep_stages))

        for i in range(len(sleep_stages)):
            label.append(int(sleep_stages[i].firstChild.data)) # '0' '1' '2' '3' '5' -> 0 1 2 3 5
        # print(label[:100])

        label = np.array(label)
        # print(np.unique(label, return_counts=True))
        label = np.where(label==5, 4, label) # turn label from 01235 to 01234
        # print(np.unique(label, return_counts=True))
        return label # ndarray
    
    def get_data(self, data_path, subject):
        sample_rate = 256
        filename = f"{data_path}0{subject+1}.edf"
        data = self.load_edf(filename,subject)
        if self.downsample is not None:
            data = mne.filter.resample(data, down=sample_rate/self.downsample, axis=1)
            sample_rate = self.downsample
        data_slice = self.slice_data(data, self.time_length, sample_rate)
        return data_slice

    def load_edf(self, filename,subject):
        raw_edf = mne.io.read_raw_edf(filename)
        if self.task=='children':
            self.target_channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1','E1-M2', 'E2-M2','O1-M2','O2-M1', 'Chin1-Chin2'] # 'E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1','Chin 3-Chin 2'  
        elif self.task=='adults':
            self.target_channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1','E1-M2', 'E2-M2','O1-M2','O2-M1', 'Chin 1-Chin 2']
        if (self.task == 'children') & (subject == 2):
            self.target_channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1','E1-M2', 'E2-M2', 'O1-M2','O2-M1','chin1-chin2']
        all_channels = raw_edf.ch_names
        # print(type(all_channels), all_channels)

        exclude = self.exclude_channels(all_channels, self.target_channels)
        raw_edf = mne.io.read_raw_edf(filename, exclude=exclude) # using exclude to avoid unnecessary resampling while reading data with different sampling rates
        # print(raw_edf.info)
        data = raw_edf.get_data()
        print('raw_edf.ch_names',raw_edf.ch_names)
        # print(data.shape) # 8*N ndarray
        return data

    def exclude_channels(self, all_channels, target_channels):
        bad_channels = ['Nasal Pressure', 'Pressure']
        all_channels = all_channels + bad_channels

        exclude = [i for i in all_channels if i not in set(target_channels)]
        # print(exclude)
        print(target_channels)
        return exclude # list

    def slice_data(self, data, time_length, sample_rate): # data(ndarray)
        sample_point = data.shape[1]
        window_size = time_length * sample_rate
        num_of_window = sample_point // window_size
        # print(num_of_window)
        trim_data = data[:,:num_of_window*window_size] # eg. 42011->42000
        data_slice = trim_data.reshape(data.shape[0], num_of_window, window_size)
        # print(data_slice.shape) # (8, 1023, 7680)
        data_slice = data_slice.transpose(1, 0, 2)
        # print(data_slice.shape) # (1023, 8, 7680)
        return data_slice

if __name__ == '__main__':
    import time
    start_time = time.time()
    # dataset = SleepDataset()
    dataset = SleepDataset(task='children', target_channels=['C4-M1'], downsample=100)
    print(time.time()-start_time)
    print('Done!')