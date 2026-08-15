import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, dataframe, image_size=128):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = float(row["has_tumor"])

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor