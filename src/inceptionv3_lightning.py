
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import inception_v3
from tqdm import tqdm
import torch
import pytorch_lightning as pl

class InceptionV3LightningModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate = 1e-4,
        num_epochs = 1,
        in_features = 2048, 
        out_features=5,
        bias=True,
        aux_logits = False,
    ):
        super().__init__()
        print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = inception_v3(pretrained=True).to(device=self.device)
        for param in self.model.parameters():
            param.requires_grad = False 

        self.model.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.aux_logits = aux_logits
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X.to(device=self.device)
        y = y.to(device=self.device)
        y_hat = self.model(X)
        loss = self.loss_criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(device=self.device)
        y = y.to(device=self.device)
        y_hat = self.model(X)
        loss = self.loss_criterion(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

    