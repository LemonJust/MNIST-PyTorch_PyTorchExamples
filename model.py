import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MnistMLP(nn.Module):
    def __init__(self,in_dims=None,out_dims=None):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_dims,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
    
    def configure_optimizer(self,train_config):
        optimizer = torch.optim.Adam(self.parameters(),lr=train_config.learning_rate,
                                     weight_decay=train_config.weight_decay,betas=train_config.betas)
        return optimizer

    def forward(self, x, targets):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))        
        logits = self.fc4(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(in_features=(16*5*5),out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)

    def configure_optimizer(self,train_config):
        optimizer = torch.optim.Adam(self.parameters(),lr=train_config.learning_rate,
                                    betas=train_config.betas,weight_decay=train_config.weight_decay)
        return optimizer

    def forward(self, x, targets):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.fc3(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits,targets)
        return logits,loss