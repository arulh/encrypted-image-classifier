import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import os

class ImageDataset(Dataset):
    def __init__(self, image_folder: str, labels_file: str):
        curr_dir = os.path.dirname(__file__)
        self.img_folder_dir = os.path.join(curr_dir, f"../data/processed/{image_folder}")
        labels_path = os.path.join(curr_dir, f"../data/processed/labels/{labels_file}")
        self.data = pd.read_csv(labels_path, header=None)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        image_name, target = self.data.iloc[idx]
        image = read_image(os.path.join(self.img_folder_dir, image_name))
        return image, target
    
if __name__ == "__main__":
    dog_images = ImageDataset("dogs", "dogs_labels.csv")
    print(dog_images[0][0])