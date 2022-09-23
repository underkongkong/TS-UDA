from util.utils import decomposition_UMAP_2D,decomposition_UMAP_3D, decomposition_UMAP_labels
from dataset.data_loaders import dataset_generator,data_generator_augment
import os
from datetime import datetime
import re



# import umap
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# digits = load_digits()
# print(digits.data.shape)
# embedding = umap.UMAP().fit_transform(digits.data)
# print(embedding.shape)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the Digits dataset')
# plt.show()

start_time = datetime.now()
save_path = '../experiments_save/figures/decomposition'
os.makedirs(save_path, exist_ok=True)
# get source and target data


target_path= '/home/brain/code/SleepStagingPaper/data_npy/isruc-sleep-3/'
source_path= '/home/brain/code/SleepStagingPaper/data_npy/sleepedf-78/'

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

source_files = source_path_[80:]
target_files = target_path_[:2]
# target_files = source_path_[8:]

data = dataset_generator(source_files,target_files)

# source_weak_dataset,source_strong_dataset,source_test_dataset = data[0]
# target_weak_dataset,target_strong_dataset,target_test_dataset = data[1]

print(len(data))
# labels = ['source_weak','source_strong','source_test','target_strong','target_weak','target_test']
labels = ['source','target']

# decomposition_UMAP_2D(data,save_path,labels)
# decomposition_UMAP_3D(data,save_path,labels)


decomposition_UMAP_labels(data,save_path,labels)

end_time = datetime.now()
print("Time Cost:",end_time-start_time)