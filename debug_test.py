import numpy as np
import os
dir= os.listdir('/home/brain/code/SleepStagingPaper/data_npy/shhs_npz/')
for subject in dir:
    path = f'/home/brain/code/SleepStagingPaper/data_npy/shhs_npz/{subject}'
    data= np.load(path)
    if data['x'].shape[2]!=4:
        print(data['x'].shape)