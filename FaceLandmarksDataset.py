import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[item, 0])
        image = io.imread(img_name)
        landmarks = self.df.iloc[item, 1:].to_numpy(dtype='float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
