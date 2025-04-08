import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchmetrics import Accuracy

import pickle

from tqdm import tqdm

class SupervisedTraining:
    def __init__(
            self,
            model : nn.Module,
            train_loader : DataLoader,
            val_loader : DataLoader,
            num_epochs : int,
            lr : float,
            criterion=nn.MSELoss(),
            optimizer=optim.Adam,
            scheduler=optim.lr_scheduler.StepLR,
            scheduler_params={'step_size': 10, 'gamma': 0.1},
            is_classification=True,
            num_classes=2,
            device='mps',
        ): 
        self.device = device

        self.model = model.to(self.device,dtype=torch.float32)
        self.criterion = criterion.to(self.device,dtype=torch.float32)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.scheduler = scheduler(self.optimizer, **scheduler_params) if scheduler is not None else None
        self.scheduler_params = scheduler_params

        self.num_epochs = num_epochs
        self.lr = lr

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.is_classification = is_classification
        self.num_classes = num_classes
        if is_classification:
            assert num_classes is not None, 'num_classes must be specified for classification tasks'

    def _get_accuracy(self, outputs, targets):     
        """
        Computes accuracy for classification tasks.
        """
        if self.num_classes != 2:
            preds = torch.argmax(outputs, dim=1)
            accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
            return accuracy(preds, targets).item(), 0, 0
        else:
            preds = (outputs > 0.5).float()
            P_detection = (torch.sum(preds * targets) / torch.sum(targets)).item()
            P_false_alarm = (torch.sum(preds * (1 - targets)) / torch.sum(1 - targets)).item()
            return (P_detection + 1 - P_false_alarm) / 2, P_detection, P_false_alarm

    def _train_epoch(self):
        """
        Performs one training epoch.
        """

        current_train_loss = 0.0
        current_accuracy = 0.0
        current_P_detection = 0.0
        current_P_false_alarm = 0.0

        self.model.train()
        for train_inputs, train_targets in self.train_loader:
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)

            self.optimizer.zero_grad()

            train_outputs = self.model(train_inputs)
            train_outputs = train_outputs if self.is_classification and self.num_classes != 2 else train_outputs.flatten()
            train_loss = self.criterion(train_outputs, train_targets)

            current_train_loss += train_loss.item()

            train_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Compute accuracy for classification tasks
            if self.is_classification:
                accuracy, P_detection, P_false_alarm = self._get_accuracy(train_outputs, train_targets)

                current_accuracy += accuracy
                current_P_detection += P_detection
                current_P_false_alarm += P_false_alarm

        return current_train_loss/len(self.train_loader), current_accuracy/len(self.train_loader), current_P_detection/len(self.train_loader), current_P_false_alarm/len(self.train_loader)
    
    def _val_epoch(self):
        """
        Performs one validation epoch.
        """

        current_val_loss = 0.0
        current_accuracy = 0.0
        current_P_detection = 0.0
        current_P_false_alarm = 0.0

        self.model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in self.val_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                val_outputs = self.model(val_inputs)
                val_outputs = val_outputs if self.is_classification and self.num_classes != 2 else val_outputs.flatten()
                val_loss = self.criterion(val_outputs, val_targets)

                current_val_loss += val_loss.item()

                # Compute accuracy for classification tasks
                if self.is_classification:
                    accuracy, P_detection, P_false_alarm = self._get_accuracy(val_outputs, val_targets)

                    current_accuracy += accuracy
                    current_P_detection += P_detection
                    current_P_false_alarm += P_false_alarm

        return current_val_loss/len(self.val_loader), current_accuracy/len(self.val_loader), current_P_detection/len(self.val_loader), current_P_false_alarm/len(self.val_loader)

    def _save_model(self, outpath):
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

        train_P_detections = []
        val_P_detections = []

        train_P_false_alarms = []
        val_P_false_alarms = []

        best_val_loss = float('inf')

        # Train for the specified number of epochs
        for epoch in tqdm(range(1,self.num_epochs+1),desc='Training'):
            # Train and validate
            train_loss, train_acc, train_P_D, train_P_F = self._train_epoch()
            val_loss, val_acc, val_P_D, val_P_F = self._val_epoch()

            # Save model if validation loss is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(f'{outpath}/model/best.pth')

            if epoch % save_training_stats_every == 0 or epoch == 1:
                tqdm.write(f'Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}')

            # Save test results and model checkpoints if necessary
            if save_model_every and epoch % save_model_every == 0 and val_loss != best_val_loss:
                self._save_model(f'{outpath}/model/epoch_{epoch:04d}.pth')

            # Save training stats
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            train_P_detections.append(train_P_D)
            val_P_detections.append(val_P_D)

            train_P_false_alarms.append(train_P_F)
            val_P_false_alarms.append(val_P_F)

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
                        
                        training_stats['train_P_detections'] = train_P_detections
                        training_stats['val_P_detections'] = val_P_detections

                        training_stats['train_P_false_alarms'] = train_P_false_alarms
                        training_stats['val_P_false_alarms'] = val_P_false_alarms

                    pickle.dump(training_stats, f)