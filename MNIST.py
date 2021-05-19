import numpy as np
import torch
import torch.nn as nn
import torchvision

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.utils.data import random_split

import matplotlib.pyplot as plt

from model import MnistMLP,MnistCNN
from trainer import TrainerConfig,Trainer
from visualize import Plot

train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

if __name__ == "__main__":
    cnn_train_configs = TrainerConfig(ckpt_path="./test.pt",max_epochs=40,learning_rate=4.67e-4,weight_decay=6.423e-4,num_workers=2)
    CNN_model = MnistCNN()
    trainer = Trainer(model=CNN_model,train_dataset=train_set,test_dataset=test_set,config=cnn_train_configs)
    model_metrics = trainer.train()

    plotter = Plot(model_metrics=model_metrics)
    plotter.plot()