import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image
import seaborn as sns 
import numpy as np
import pandas as pd
import time

class Plot():

    def __init__(self,model_metrics):
        self.model = model_metrics.model
        self.epochs = model_metrics.epochs
        self.train_losses = model_metrics.train_losses
        self.smoothed_train_losses = model_metrics.smoothed_train_losses
        self.train_accuracies = model_metrics.train_accuracies
        self.smoothed_train_accuracies = model_metrics.smoothed_train_accuracies
        self.test_losses = model_metrics.test_losses
        self.test_accuracies = model_metrics.test_accuracies

    def plot_train_loss(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        

        plt.plot(df["train_losses"],label="Train Loss",alpha=0.3,color="orange",marker="o")
        plt.plot(df["smoothed_train_losses"],label="Avg Train Loss",alpha=0.9,color="red")

        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)

        labels = np.arange(1,len(df["train_losses"])+1,1e3)
        xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
        xlabels[0] = '0'
        locs =  np.arange(1,len(df["train_losses"])+1,1e3).astype(int)
        plt.xticks(ticks=locs,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Training Metrics (Train Loss)",pad=20,fontsize=15)
        plt.xlabel("Iterations",fontsize=15,labelpad=15)
        plt.ylabel("Train Loss",fontsize=15,labelpad=15)
        plt.show()

    def plot_train_accuracies(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.plot(df["train_accuracies"],label="Train Accuracy",alpha=0.2,color="orange",marker="o")
        plt.plot(df["smoothed_train_accuracies"],label="Avg Train Accuracy",alpha=0.9,color="red")
        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
        labels = np.arange(1,len(df["train_accuracies"])+1,1e3)
        xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
        xlabels[0] = '0'
        locs = np.arange(1,len(df["train_accuracies"])+1,1e3).astype(int)
        plt.xticks(ticks=locs,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Training Metrics (Train Accuracy)",pad=20,fontsize=15)
        plt.xlabel("Iterations",fontsize=15,labelpad=15)
        plt.ylabel("Train Accuracy",fontsize=15,labelpad=15)
        plt.show()

    def plot_test_accuracies(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.plot(df["smoothed_test_accuracies"],label="Avg Test Accuracy",alpha=0.8,color="red")
        plt.plot(df["test_accuracies"],label="Test Accuracy",alpha=0.8,color="darkgreen")

        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
        
        xlabels = ['{:,.0f}'.format(x) for x in range(1,len(df["test_accuracies"])+1,1)]
        locs = np.arange(0,len(df["test_accuracies"])).astype(int)
        plt.xticks(ticks=locs ,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Valuation Metrics (Accuracy)",pad=20,fontsize=15)
        plt.xlabel("Epochs",fontsize=15,labelpad=15)
        plt.ylabel("Accuracy",fontsize=15,labelpad=15)
        plt.show()

    def plot_test_losses(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.plot(df["smoothed_test_losses"],label="Avg Test Loss",alpha=0.8,color="red")
        plt.plot(df["test_losses"],label="Test Loss",alpha=0.8,color="darkgreen")
        
        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
        
        xlabels = ['{:,.0f}'.format(x) for x in range(1,len(df["test_accuracies"])+1,1)]
        locs = np.arange(0,len(df["test_accuracies"])).astype(int)
        plt.xticks(ticks=locs ,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Valuation Metrics (Loss)",pad=20,fontsize=15)
        plt.xlabel("Epochs",fontsize=15,labelpad=15)
        plt.ylabel("Loss",fontsize=15,labelpad=15)
        plt.show()

    def plot_avg_metrics(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.plot(df["epoch_losses"],label="Avg Train Loss",alpha=0.3,color="red",marker="o")
        plt.plot(df["smoothed_test_losses"],label="Avg Test Loss",alpha=0.3,color="darkgreen",marker="o")
        
        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
        
        xlabels = ['{:,.0f}'.format(x) for x in range(1,len(df["test_accuracies"])+1,1)]
        locs = np.arange(0,len(df["epoch_losses"])).astype(int)
        plt.xticks(ticks=locs ,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Valuation Metrics (Loss)",pad=20,fontsize=15)
        plt.xlabel("Epochs",fontsize=15,labelpad=15)
        plt.ylabel("Loss",fontsize=15,labelpad=15)
        plt.show()

    def plot_avg(self,df):
        sns.set()
        plt.style.use('seaborn-ticks')
        plt.figure(figsize=(15,20))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.plot(df["smoothed_test_accuracies"],label="Avg Test Accuracy",alpha=0.3,color="darkgreen",marker="o")
        plt.plot(df["epoch_accuracies"],label="Avg Train Accuracy",alpha=0.3,color="red",marker="o")
        
        yticks = plt.yticks()
        for y_locs in yticks[0][1:]:
            plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
        
        xlabels = ['{:,.0f}'.format(x) for x in range(1,len(df["epoch_accuracies"])+1,1)]
        locs = np.arange(0,len(df["epoch_accuracies"])).astype(int)
        plt.xticks(ticks=locs ,labels=xlabels)
        plt.legend(loc=0,prop={'size':10},)
        plt.title("Model Valuation Metrics (Accuracy)",pad=20,fontsize=15)
        plt.xlabel("Epochs",fontsize=15,labelpad=15)
        plt.ylabel("Loss",fontsize=15,labelpad=15)
        plt.show()

    def plot_model_weights(self): 
        kernels = self.model.conv1.weight.detach().clone()
        kernels = kernels.to("cpu")
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        # filter_img = make_grid(kernels, nrow = 6)
        # change ordering since matplotlib requires images to 
        # be (H, W, C)
        # plt.imshow(filter_img.permute(1, 2, 0))
        img = save_image(kernels,'kernels'+str(self.epochs)+".png",nrow=6)

    def plot(self):
        train_losses = pd.Series(self.train_losses,name="train_losses")
        train_accuracies = pd.Series(self.train_accuracies,name="train_accuracies")
        smoothed_train_losses = pd.Series(self.smoothed_train_losses,name="smoothed_train_losses")
        smoothed_train_accuracies = pd.Series(self.smoothed_train_accuracies,name="smoothed_train_accuracies")

        test_losses = pd.Series(self.test_losses,name="test_losses")
        test_accuracies = pd.Series(self.test_accuracies,name="test_accuracies")
        
        smoothed_test_losses = pd.Series(np.array([np.mean(test_losses[:idx_slice+1]) for idx_slice in range(len(test_losses))]),name="smoothed_test_losses")
        smoothed_test_accuracies = pd.Series(np.array([np.mean(test_accuracies[:idx_slice+1]) for idx_slice in range(len(test_accuracies))]),name="smoothed_test_accuracies")
        epochs_index = np.arange(938,len(test_losses)*938,937)
        
        epoch_accuracies = pd.Series(np.array([smoothed_train_accuracies[x] for x in epochs_index]),name="epoch_accuracies")
        epoch_losses = pd.Series(np.array([smoothed_train_losses[x] for x in epochs_index]),name="epoch_losses")

        Train_df = pd.concat([train_losses,train_accuracies,smoothed_train_losses,smoothed_train_accuracies],axis=1)
        Test_df = pd.concat([test_losses,test_accuracies,epoch_accuracies,epoch_losses,smoothed_test_losses,smoothed_test_accuracies],axis=1)

        self.plot_train_loss(Train_df)
        self.plot_train_accuracies(Train_df)
        self.plot_test_losses(Test_df)
        self.plot_test_accuracies(Test_df)
        self.plot_avg_metrics(Test_df)
        self.plot_avg(Test_df)
        self.plot_model_weights()