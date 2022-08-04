from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os

class RetinaDataset(Dataset):
    def __init__(self, imagepath="data/archive/resized_train_cropped/resized_train_cropped/", total=None):
        transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
        print(os.getcwd())
        self.df = pd.read_csv("data/trainLabels_cropped.csv")
        
        if (total is not None):
            self.df = self.df[:total]
        
        self.transform = transform
        
        self.imagepath = imagepath
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.imagepath, self.df.iloc[index].image +".jpeg")
        img = Image.open(img_path)

        # if(self.transform):
        #     # serialize the transformed image!
        #     img = self.transform(img)
        
        # serialize the tensor!
        # return img, torch.tensor(self.df.iloc[index].level)

         # from tensor file!
        tensor_path = os.path.join(self.imagepath, + "tensor/" + self.df.iloc[index].image +".pt")
        return img, torch.load(tensor_path)

