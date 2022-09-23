from dataset.data import get_dataloaders
import argparse
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
from util.utils import _logger,copy_Files
from datetime import datetime

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
        self.features_len = 127
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

start_time = datetime.now()
# get source and target data
# data = get_dataloaders("./", batch_size_source=32, workers=2)

# source_path = f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/'
# target_path = f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/'
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')

parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')

parser.add_argument('--encoder', default='MMASleepNet_EEG', type=str,
                    help='Encoder model name')

parser.add_argument('--seed', default=0, type=int,
                    help='seed value')

parser.add_argument('--logs_save_dir', default='../experiments_save/', type=str,
                    help='saving directory')

parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')

parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--target_path', default=f'/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3-reg/', type=str,
                    help='Target data path')

parser.add_argument('--source_path', default=f'/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78-reg/', type=str,
                    help='Target data path')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
encoder = args.encoder
SEED = args.seed
setup_seed(SEED)

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description+f"_seed_{SEED}")
model_save_dir = os.path.join(experiment_log_dir,'model_save')
log_dir = os.path.join(experiment_log_dir,'logs')
os.makedirs(experiment_log_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 备份文件
copy_Files(os.path.join(logs_save_dir, experiment_description, run_description+f"_seed_{SEED}"),home_dir,encoder_name=encoder)

target_path= args.target_path
source_path= args.source_path

log_file_name = os.path.join(log_dir, f"log_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Source Path: {source_path}')
logger.debug(f'Target Path:  {target_path}')
logger.debug(f'Model:    {encoder}')
logger.debug("=" * 45)

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

source_files = source_path_[150:]
target_files = target_path_
# target_files = source_path_[8:]

data = data_generator_augment(source_files,target_files,batch_size=32,workers=2)
source_dataloader_train_weak, source_dataloader_train_strong, source_dataloader_test = data[0]
target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]
logger.debug("Data loaded ...")

configs = Config()
# instantiate the network
n_classes = 5
# encoder = model.base_Model(configs)
# encoder = model.MMASleepNet_eegeogemg_plus_encoder(configs)
encoder = eval(f'model.{encoder}(configs)')
classifier = model.Classifier(configs.features_len,n_classes=n_classes)
# print('classifier',classifier)#,classifier.shape)

# instantiate AdaMatch algorithm and setup hyperparameters
adamatch = model.Damatch(encoder, classifier)
hparams = adamatch_hyperparams()
epochs = 5 # my implementations uses early stopping



# train the model
adamatch.train(source_dataloader_train_weak, source_dataloader_train_strong,
               target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
               epochs, hparams, model_save_dir+'/checkpoint.pt')

# evaluate the model
adamatch.plot_metrics(experiment_log_dir)

# returns accuracy on the test set
print(f"accuracy on test set = {adamatch.evaluate(target_dataloader_test)}")

# returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
adamatch.plot_cm_roc(target_dataloader_test,experiment_log_dir)

