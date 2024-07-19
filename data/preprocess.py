import torch 
import numpy as np 
import pickle 
from sklearn.preprocessing import MinMaxScaler

def process_yolo_output(sample, normalize=True):
    scaler = MinMaxScaler()

    sample = torch.stack([torch.as_tensor(x, dtype = float) for x in sample])

    if normalize:
        sample = scaler.fit_transform(sample)

    return sample

def main():
    with open("D:\Betafinder\Betafinder/betafinder_dataset/train\input/train1.pkl", "rb") as file:
        features = pickle.load(file)
        file.close()

    res = process_yolo_output(features[0])
    print(res)
    

if __name__ == '__main__':
    main()