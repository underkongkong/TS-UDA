from dataset.data import get_dataloaders
# from model.network import Encoder, Classifier
from dataset.hyperparameters import adamatch_hyperparams
# from model.adamatch import Adamatch
import torch
import numpy as np
import random
from dataset.data_loaders import data_generator_augment
import os
import model
import re
class Config():
    def __init__(self,  
                # channels
                ):
        self.input_channels = 1
        self.final_out_channels = 128
        self.num_classes = 5
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 128
        self.afr_reduced_cnn_size=2
        self.d_model = 48
        self.inplanes=2
        self.nhead=4
        self.num_layers=1

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
# get source and target data
# data = get_dataloaders("./", batch_size_source=32, workers=2)

# source_path = f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/'
# target_path = f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/'
target_path= f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/'
source_path= f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/'

source_files = os.listdir(source_path)
source_files.sort(key=lambda x: int(str(re.findall("\d+", x)[0])))
target_files = os.listdir(target_path)
target_files.sort(key=lambda x: int(str(re.findall("\d+", x)[0])))
source_path_=[]
target_path_ = []
for file in source_files:
    source_path_.append(source_path+file)
for file in target_files:
    target_path_.append(target_path+file)
# print(path)
print("source_files_len:",len(source_path_))
print("target_files_len:",len(target_path_))

source_files = source_path_[0:20]
target_files = target_path_[0:4]
# target_files = source_path_[8:]

data = data_generator_augment(source_files,target_files,batch_size=32,workers=2)
source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]
configs = Config()
# instantiate the network
n_classes = 5
# encoder = model.base_Model(configs)
# encoder = model.MMASleepNet_eegeogemg_plus_encoder(configs)
encoder = model.MMASleepNet_eeg_plus_encoder(configs)
classifier = model.Classifier(configs.features_len,n_classes=n_classes)
print('classifier',classifier)#,classifier.shape)

# instantiate AdaMatch algorithm and setup hyperparameters
adamatch = model.Adamatch(encoder, classifier)
hparams = adamatch_hyperparams()
epochs = 500 # my implementations uses early stopping
# save_path = "./adamatch_checkpoint.pt"
save_path = "/home/lyy/UDA-sleep/ckpt/S_isruc-sleep-1_T_sleep-edf-20"#/{self.model.init_time}/"

# train the model
adamatch.train(source_dataloader_train_weak, source_dataloader_train_strong,
               target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
               epochs, hparams, save_path)

# evaluate the model
adamatch.plot_metrics()

# returns accuracy on the test set
print(f"accuracy on test set = {adamatch.evaluate(target_dataloader_test)}")

# returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
adamatch.plot_cm_roc(target_dataloader_test)