import numpy as np
import torch
from torch.utils.data import Dataset

class SWaTWindowDataset(Dataset):
    def __init__(self, npz_path):
        x_list, y_list, ts_list = [], [], []
        for pth in npz_path:
            d = np.load(pth, allow_pickle=True)
            x_list.append(d["x"])
            y_list.append(d["y"])
            ts_list.append(d["ts"])
            sensors_name = d["sensors_name"]
            mean = d["mean"]
            std = d["std"]
        self.x = torch.tensor(np.concatenate(x_list), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list), dtype=torch.long)
        self.ts = np.concatenate(ts_list)
        self.sensors_name = sensors_name
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.ts[i]