from  inceptionv3_lightning import *
import ssl
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer
from argparse import ArgumentParser


from retina_dataset import RetinaDataset

ssl._create_default_https_context = ssl._create_unverified_context

def get_cropped_data(relative_path = "../input/diabetic-retinopathy-resized/trainLabels_cropped.csv", count=5000):
    return pd.read_csv(relative_path)[:count]

# train.py
def main(args):
    batch_size =  args.batch_size if(args and  args.batch_size) else 32
    num_devices = args.num_devices if(args and args.num_devices) else 8
    num_nodes = args.num_nodes if(args and args.num_nodes) else 4

    model = InceptionV3LightningModel(args)
    train_dataset = RetinaDataset(total=5000)
    train, val = random_split(train_dataset, [4500, 500])
    
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True)

    trainer = Trainer(accelerator="gpu", devices=num_devices, num_nodes=num_nodes, strategy="ddp")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--num_devices", default=None)
    parser.add_argument("--num_nodes", default=None)
    args = parser.parse_args()

    main(args)