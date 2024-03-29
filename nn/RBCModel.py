'''
source: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta


class RBCModel(nn.Module, metaclass=ABCMeta):
    def training_step(self, criterion,  data, labels):
        latent_pi, latent_vf , _= self._get_latent(data)   # Generate predictions
        prediction = self.action_net(latent_pi)
        loss = criterion(prediction, labels) # Calculate loss
        _, preds = torch.max(prediction, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds)) # Calculate accuracy
        return {'train_loss':loss, 'train_acc': acc}

    def validation_step(self, criterion, data, labels):
        latent_pi, latent_vf , _= self._get_latent(data)   # Generate predictions
        prediction = self.action_net(latent_pi)
        loss = criterion(prediction, labels)   # Calculate loss
        _, preds = torch.max(prediction, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds)) # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
