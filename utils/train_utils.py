import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

import pickle

from tqdm import tqdm

class SupervisedTraining:
    def __init__(
            self,
            model : nn.Module,
            trainloader : DataLoader,
            valloader : DataLoader,
            num_epochs : int,
            lr : float,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam,
            scheduler=None,
            is_classification=True,
            device='mps',
    ): 
        self.device = device

        self.model = model.to(self.device,dtype=torch.float32)
        self.criterion = criterion.to(self.device,dtype=torch.float32)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.lr = lr

        self.trainloader = trainloader
        self.valloader = valloader

        self.is_classification = is_classification

    def get_accuracy(self, outputs, targets):     
        """
        Computes accuracy for classification tasks.
        """

        preds = torch.argmax(outputs, dim=1)
        return Accuracy(preds, targets)

    def train_epoch(self):
        """
        Performs one training epoch.
        """

        current_train_loss = 0.0
        accuracy = 0.0

        self.model.train()
        for train_inputs, train_targets in self.trainloader:
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)

            self.optimizer.zero_grad()

            train_outputs = self.model(train_inputs)
            train_loss = self.criterion(train_outputs if self.is_classification else train_outputs.flatten(), train_targets)

            current_train_loss += train_loss.item()

            train_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Compute accuracy for classification tasks
            if self.is_classification:
                accuracy += self.get_accuracy(train_outputs, train_targets)

        return current_train_loss/len(self.trainloader), accuracy/len(self.trainloader)
    
    def val_epoch(self):
        """
        Performs one validation epoch.
        """

        current_val_loss = 0.0
        accuracy = 0.0

        self.model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in self.valloader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                val_outputs = self.model(val_inputs)
                val_loss = self.criterion(val_outputs if  self.is_classification else val_outputs.flatten(), val_targets)

                current_val_loss += val_loss.item()

                # Compute accuracy for classification tasks
                if self.is_classification:
                    accuracy += self.get_accuracy(val_outputs, val_targets)

        return current_val_loss/len(self.valloader), accuracy/len(self.valloader)

    def save_model(self, outpath):
        """
        Saves the model and optimizer state.
        """

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, outpath)

    def train(self, save_training_stats_every=10, save_model_every=None, outpath='training_result'):
        """
        Trains the model for the specified number of epochs and optionally saves training results and model checkpoints.
        """

        # Create directories if necessary
        if save_training_stats_every or save_model_every:
            assert outpath is not None, 'outpath must be specified when save_training_stats_every or save_model_every is specified'

            if not os.path.exists(os.path.join(outpath,'model')):
                os.makedirs(os.path.join(outpath,'model'))

        # Initialize training stats
        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        best_val_loss = float('inf')

        # Train for the specified number of epochs
        for epoch in tqdm(range(1,self.num_epochs+1),desc='Training'):
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()

            # Save model if validation loss is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'{outpath}/model/best.pth')

            if epoch % save_training_stats_every == 0:
                tqdm.write(f'Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}')

            # Save test results and model checkpoints if necessary
            if save_model_every and epoch % save_model_every == 0 and val_loss != best_val_loss:
                self.save_model(f'{outpath}/model/epoch_{epoch:04d}.pth')

            # Save training stats
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if save_training_stats_every and epoch % save_training_stats_every == 0:
                with open(f'{outpath}/training_stats.pkl', 'wb') as f:
                    training_stats = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                    }

                    # Save accuracies for classification tasks
                    if self.is_classification:
                        training_stats['train_accuracies'] = train_accuracies
                        training_stats['val_accuracies'] = val_accuracies

                    pickle.dump(training_stats, f)