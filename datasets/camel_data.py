import random
import torch
import pandas as pd
from pathlib import Path
import torch.utils.data as data

class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        self.state = state

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        self.shuffle = self.dataset_cfg.data_shuffle

        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()

        self.text_data_dir = self.dataset_cfg.text_data_dir
        self.max_length = self.find_max_length()

    def find_max_length(self):
        max_length = 0
        for slide_id in self.data:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            if full_path.exists():
                features = torch.load(full_path)
                max_length = max(max_length, features.shape[0])
        return max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        features = torch.load(full_path)

        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
        
        if features.shape[0] < self.max_length:
            pad = torch.zeros((self.max_length - features.shape[0], 1024))
            features = torch.cat((features, pad), dim=0)
        else:
            features = features[:self.max_length]

        if self.state == "test":
           text = None
        else:
            text_path = Path(self.text_data_dir) / f'{slide_id}.txt'
            with open(text_path, 'r') as f:
                text = f.read()

        return features, label, text

