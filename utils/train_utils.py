import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

import pickle

from tqdm import tqdm

class Train:
    def __init__(
            self,
            model : nn.Module,
            trainloader : DataLoader,
            valloader : DataLoader,
            num_epochs : int,
            lr : float,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,
            scheduler=None,
            accuracy=None,
            device='mps',
    ): 
        """
        Initializes the Train class with required parameters.

        Args:
            model (torch.nn.Module): The model to be trained.
            trainloader (DataLoader): DataLoader for the training dataset.
            valloader (DataLoader): DataLoader for the validation dataset.
            num_epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
            criterion (torch.nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
            optimizer (torch.optim.Optimizer): Optimizer class. Defaults to Adam.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            accuracy (callable, optional): Accuracy metric function. Defaults to None.
            device (str): Device to run training ('cpu', 'cuda', 'mps'). Defaults to 'mps'.
        """

        self.device = device

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.lr = lr

        self.trainloader = trainloader
        self.valloader = valloader

        self.accuracy = accuracy

    def get_accuracy(self, outputs, targets):     
        """
        Computes accuracy using the provided accuracy metric.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            float: Accuracy score.
        """

        preds = torch.argmax(outputs, dim=1)
        return self.accuracy(preds, targets)

    def train_epoch(self):
        """
        Performs one training epoch.

        Returns:
            tuple: Average training loss and accuracy (if accuracy is defined) for the epoch.
        """

        current_train_loss = 0.0
        accuracy = 0.0

        for train_inputs, train_targets in tqdm(self.trainloader):
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)

            self.optimizer.zero_grad()

            train_outputs = self.model(train_inputs)
            train_loss = self.criterion(train_outputs.flatten(), train_targets)
            train_loss.backward()
            self.optimizer.step()

            current_train_loss += train_loss.item()
            if accuracy is not None:
                accuracy += self.get_accuracy(train_outputs, train_targets)

        return current_train_loss/len(self.trainloader), accuracy/len(self.trainloader)
    
    def val_epoch(self):
        """
        Performs one validation epoch.

        Returns:
            tuple: Average validation loss and accuracy (if accuracy is defined) for the epoch.
        """

        current_val_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for val_inputs, val_targets in tqdm(self.valloader):
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(val_outputs.flatten(), val_targets)

                current_val_loss += val_loss.item()

                if accuracy is not None:
                    accuracy += self.get_accuracy(val_outputs, val_targets)

        return current_val_loss/len(self.valloader), accuracy/len(self.valloader)
    
    def save_model(self,out_path):
        """
        Saves the model and optimizer state.

        Args:
            out_path (str): Path to save the model checkpoint.
        """

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, out_path)

    def train(self,train_results_path=None,save_every=None,model_path=None):
        """
        Trains the model for the specified number of epochs and optionally saves training results and model checkpoints.

        Args:
            train_results_path (str, optional): Path to save the training results (losses and accuracies) as a pickle file. Defaults to None.
            save_every (int, optional): Interval (in epochs) to save the model checkpoint. Defaults to None.
            model_path (str, optional): Directory path where model checkpoints will be saved. Required if save_every is specified.
        """

        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss, train_acc = self.train_epoch()

            self.model.eval()
            val_loss, val_acc = self.val_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if save_every and epoch % save_every == 0:
                self.save_model(f'{model_path}/epoch_{epoch}.pth')

            print(f'Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        if train_results_path:
            with open(train_results_path, 'wb') as f:
                pickle.dump({
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                }, f)