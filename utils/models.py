import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from sklearn.feature_selection import f_classif, SelectKBest
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import f1_score, auroc, precision, recall, accuracy

# Define the model
class GeneModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hl1_size, hl2_size, hl3_size, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hl1_size: integer, size of the first hidden layer
            hl2_size: integer, size of the second hidden layer
            hl3_size: integer, size of the third hidden layer
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        hidden_layers = [hl1_size, hl2_size, hl3_size]

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hl1_size)])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hl3_size, output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return torch.sigmoid(x)

class DeepGeneModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, 4000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4000),
            nn.Linear(4000, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)

class DeepctModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, 4000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(4000),
            nn.Linear(4000, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)


class DeepRadiomicsModel(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        feat_shape = 450
        if param.image_channels == 9:
            feat_shape = 1350
            weight = self.resnet.conv1.weight.clone()
            self.resnet.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = weight
                self.resnet.conv1.weight[:, 3:6] = weight
                self.resnet.conv1.weight[:, 6:] = weight
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc_frames = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 12)
        )
        self.fc_radiomics = nn.Sequential(
            nn.Linear(feat_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 12)
        )
        self.main = nn.Sequential(
            nn.Linear(24, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 250),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 74)
        )
        
    def forward(self, x_radiomics, x_frames):
        x_frames = self.resnet(x_frames)
        x_frames = self.fc_frames(x_frames.view(x_frames.size(0),-1))
        x_radiomics = self.fc_radiomics(x_radiomics)
        x = torch.cat((x_radiomics, x_frames), dim=1)
        x = self.main(x)
        return x

# create an encoder for the gene expression data
class GeneEncoder(torch.nn.Module):
    def __init__(self, input_size, latent_size) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size // 2, input_size // 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size // 3, latent_size)
        )
    
    def forward(self, x):
        return self.main(x)
   
# create decoder for the gene expression data
class GeneDecoder(torch.nn.Module):
    def __init__(self, latent_size, output_size) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, output_size // 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_size // 3, output_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.main(x)


class GeneAE(pl.LightningModule):
    def __init__(self, config):
        super(GeneAE, self).__init__()
        self.config = config
        self.pretraining = True
        self.input_size = config['input_size']
        self.latent_size = config['latent_size']
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_size // 2, self.input_size // 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_size // 3, self.latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.input_size // 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_size // 3, self.input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_size // 2, self.input_size)
        )
        self.downstream = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size // 2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.latent_size // 2, self.latent_size // 3),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.latent_size // 3, 1),
            nn.Sigmoid()
        )
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.pretrain_epochs = config['pretrain_epochs']
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        # self.save_hyperparameters('input_size', 'latent_size', 'learning_rate', 'batch_size', 'pretrain_epochs')



    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True)
        y_pred = self.downstream(z)
        down_loss = F.binary_cross_entropy(y_pred, y)
        y_pred = torch.round(y_pred)
        y = y.long()
        acc = accuracy(y_pred, y)
        self.log("train_down_loss", down_loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True)
        if self.pretraining:
            return recon_loss
        else:
            return down_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x)
        y_pred = self.downstream(z)
        down_loss = F.binary_cross_entropy(y_pred, y)
        y_pred = torch.round(y_pred)
        y = y.long()
        acc = accuracy(y_pred, y)
        prec = precision(y_pred, y)
        rec = recall(y_pred, y)
        f1 = f1_score(y_pred, y)
        auc = auroc(y_pred, y, num_classes=1)
        return {
            'val_recon_loss': recon_loss,
            "val_down_loss": down_loss,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_auc": auc
        }


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_down_loss = torch.stack([x["val_down_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        avg_prec = torch.stack([x["val_precision"] for x in outputs]).mean()
        avg_rec = torch.stack([x["val_recall"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()
        avg_auc = torch.stack([x["val_auc"] for x in outputs]).mean()
        self.log("val_recon_loss", avg_loss)
        self.log("val_down_loss", avg_down_loss)
        self.log("val_accuracy", avg_acc)
        self.log("val_precision", avg_prec)
        self.log("val_recall", avg_rec)
        self.log("val_f1", avg_f1)
        self.log("val_auc", avg_auc)

        if self.current_epoch < self.config["pretrain_epochs"]:
            self.pretraining = True
        else:
            self.pretraining = False

        


class GeneDown(pl.LightningModule):
    def __init__(self, latent_size, config):
        super().__init__()
        self.config = config
        self.main = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_size // 2, latent_size // 3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_size // 3, 1)
        )
    
    def forward(self, x):
        return self.main(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters, lr=self.config["learning_rate"], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
    
    def training_step(self, batch, batch_idx):
        z, y = batch
        x_recon = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        self.log('train_recon_loss', recon_loss)
        return recon_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        self.log('val_recon_loss', recon_loss)