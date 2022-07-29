
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
        num_epochs = 1,
        in_features = 2048, 
        out_features=5,
        bias=True,
        aux_logits = False,
    ):
        super().__init__()
        print(f'is cuda available: {torch.cuda.is_available()}')
        self._model = inception_v3(pretrained=True)
        for param in self._model.parameters():
            param.requires_grad = False 
        # Be careful not to overwrite `pl.LightningModule` attributes such as `self.model`.
        self._model.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self._num_epochs = num_epochs
        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias
        self._aux_logits = aux_logits
        self._loss_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self._model(x)
    
    def get(self, item):
        return self.__dict__[item]

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        (y_hat, _) = self._model(X)
        loss = self._loss_criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self._model(X)
        loss = self._loss_criterion(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        # self.log("val_acc", self._model.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr = 1e-4)

    