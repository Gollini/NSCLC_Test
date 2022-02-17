import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import pandas as pd
import os
from utils.mf_extraction import manual_feat_extraction

# create custom dataset and implement abstract methods
class Dataset(torch.utils.data.Dataset):
    def __init__(self, param):
        self.data_dir = param.data_dir
        self.param = param
        # read gene_exp.csv file from data_dir
        self.gene_exp = pd.read_csv(self.data_dir + '/gene_exp.csv', header=0, index_col=0)
        # read labels.csv file from data_dir
        self.labels = pd.read_csv(self.data_dir + '/labels_144.csv', header=0, index_col=0)
        self.radiomics_path = os.path.join(self.data_dir, 'ROIs_3frames', str(param.image_channels) + 'channels', 'resized_' + str(param.image_size))
        self.ct_path = os.path.join(self.data_dir, 'CT')
        if param.load_manual_features:
            self.radiomics = np.load(self.data_dir + '/mf_array.npy')
        else:
            self.radiomics = manual_feat_extraction(self.radiomics_path)
        # convert radiomics features to pandas dataframe with index as index of self.labels
        self.radiomics = pd.DataFrame(self.radiomics, index=self.labels.index)
        
    def __len__(self):
        return len(self.gene_exp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pid = self.labels.index[idx]

        # load frames of patient at index "idx" (self.labels.index[idx] is the ID of the patient at index "idx")
        if self.param.image_channels == 3:
            file = self.radiomics_path + '/' + pid + '_roi_axial.npy'
        elif self.param.image_channels == 9:
            file = self.radiomics_path + '/' + pid + '_roi_stacked.npy'

        # load numpy array at file
        frames = np.load(file)
        frames = np.transpose(frames, (2, 0, 1))

        # get gene expression data
        gene_exp = self.gene_exp.loc[pid, :].values
        # get labels
        label = self.labels.loc[pid, :].values
        # get radiomics features
        radiomics = self.radiomics.loc[pid, :].values

        # change the line below to return radiomics features and 9 frames too
        return radiomics, frames, gene_exp, label

# dataset = Dataset('data/')
# sample = dataset[0]