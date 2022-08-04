import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
file_df = pd.read_csv("data/trainLabels_cropped.csv")
relative_path = 'data/archive/resized_train_cropped/resized_train_cropped/'

file_df['full_path'] = relative_path + file_df['image'] + ".jpeg"
print(file_df.head())

for i, row in file_df.iterrows():   
    full_path = row['full_path']
    file_name = row['image']
    img = Image.open(full_path)
    tensor = transform(img)
    torch.save(tensor, relative_path + 'tensor/' + file_name + '.pt')

