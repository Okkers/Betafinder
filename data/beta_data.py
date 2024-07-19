from torch.utils.data import Dataset
import torch
import pandas as pd
import pickle
import numpy as np 

class Beta_data(Dataset):
    def __init__(self, feat, label, src_vocab_size, tgt_vocab_size, transform = None):

        self.feat = feat 

        if isinstance(label, pd.DataFrame):
            self.label = label.to_numpy()
        else:
            self.label = label 

        self.transform = transform

    def __len__(self):
        return len(self.feat)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"route": self.feat[idx], "beta": self.label[idx]}
        return sample
    
def main():
    with open("D:\Betafinder\Betafinder/betafinder_dataset/train\input/train1.pkl", "rb") as file:
        features = pickle.load(file)
        file.close()

    labels = pd.read_csv("D:\Betafinder\Betafinder/betafinder_dataset/train\label/b1.csv", header = None)
    data = Beta_data(features, labels)

    for indx, item in enumerate(data):
        print(indx+1)
        print(item["route"], item["beta"])
        print("\n")

if __name__ == '__main__':
    main()