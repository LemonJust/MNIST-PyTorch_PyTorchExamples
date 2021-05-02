import numpy as np
import logging

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from visualize import Plot

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs=10
    batch_size=64
    learning_rate=3e-4
    betas=(0.9,0.995)
    weight_decay=5e-4
    ckpt_path="./Model.pt"
    num_workers=0
    shuffle=True
    pin_memory=True

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

class Model_Metrics:
    train_losses = []
    smoothed_train_losses = []
    train_accuracies = []
    smoothed_train_accuracies = []
    test_losses = []
    test_accuracies = []
    model = None
    epochs = 0

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = "cpu"
        
        self.train_losses = []
        self.smoothed_train_losses = []
        self.train_accuracies = []
        self.smoothed_train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        logger.info("Saving %s",self.config.ckpt_path)
        torch.save(raw_model.state_dict(),self.config.ckpt_path)

    def train(self):
        model,config = self.model,self.config
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        optimizer = raw_model.configure_optimizer(config)

        def run_epoch(split):
            is_train = split=="train"
            if is_train:
                model.train()
            else:
                model.eval()
            data = self.train_dataset if is_train else self.test_dataset

            loader = DataLoader(data,batch_size=config.batch_size,
                                num_workers=config.num_workers,shuffle=config.shuffle,
                                pin_memory=config.pin_memory)
            
            losses = []
            accuracies = []
            num_samples = 0
            correct=0
            pbar = tqdm(enumerate(loader),total=len(loader)) if is_train else enumerate(loader)
            for it, (images,labels) in pbar:
                #place the data on the correct devices before training
                images = images.to(self.device)
                labels = labels.to(self.device)
                num_samples += labels.size(0)
                #forward the model
                with torch.set_grad_enabled(is_train):
                    logits,loss = model(images,labels)
                    loss = loss.mean()
                    losses.append(loss.item())
                
                with torch.no_grad():
                    pred = torch.argmax(logits,dim=1)
                    correct+=pred.eq(labels).sum().item()
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1} iter {it+1}: train_loss - {np.mean(losses)} train_acc - {correct/num_samples} lr - {config.learning_rate}")
                    
                    accuracies.append(correct/num_samples)
                    self.train_losses.append(loss.item())
                    self.smoothed_train_losses.append(np.mean(losses))
                    self.train_accuracies.append(correct/num_samples)
                    self.smoothed_train_accuracies.append(np.mean(self.train_accuracies))
                    
            if not is_train:
                test_loss = np.mean(losses)
                logger.info(f"Epoch {epoch+1} : Test Loss - {test_loss} | Test Accuracy - {correct/num_samples}")
                print(f"\nEpoch {epoch+1} : Test Loss - {test_loss} | Test Accuracy - {correct/num_samples}\n")
                self.test_losses.append(test_loss)
                self.test_accuracies.append(correct/num_samples)
                return test_loss

        best_loss = float('inf')
        test_loss = float('inf')

        for epoch in range(self.config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            
            good_model = self.test_dataset is not None or test_loss<best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

        model_metrics = Model_Metrics(train_losses=self.train_losses,
                                      smoothed_train_losses=self.smoothed_train_losses,
                                      train_accuracies=self.train_accuracies,
                                      smoothed_train_accuracies=self.smoothed_train_accuracies,
                                      test_losses=self.test_losses,
                                      test_accuracies=self.test_accuracies,
                                      model=self.model,
                                      epochs = self.config.max_epochs
                                    )
        return model_metrics