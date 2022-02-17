import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch import optim
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.append('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import KFold
from utils.models import DeepRadiomicsModel
from utils.dataset import Dataset

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='./data')
argparser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
argparser.add_argument('--exp_id', type=str, default='./ig_100')
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--one_fold', action='store_true')
argparser.add_argument('--fold', type=int, default=4)
argparser.add_argument('--n_splits', type=int, default=5)
argparser.add_argument('--seed', type=int, default=42)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--epochs', type=int, default=10000)
argparser.add_argument('--image_channels', type=int, default=3, help='options: 3, 9')
argparser.add_argument('--image_size', type=int, default=224, help='options: 112, 224')
argparser.add_argument('--load_manual_features', action='store_true')
param = argparser.parse_args()

# set criterion as loss for regression
exp_loss = nn.CrossEntropyLoss()
results = {}

# set up k-fold cross validation
kf = KFold(n_splits=param.n_splits, shuffle=True, random_state=param.seed)

train_metrics = pd.DataFrame(columns=["epoch", "fold", "accuracy", "roc_auc", "precision", "recall", "f1"])
test_metrics = pd.DataFrame(columns=["epoch", "fold", "accuracy", "roc_auc", "precision", "recall", "f1"])

dataset = Dataset(param)

print('--------------------------------')
# GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA Available')
else:
    device = torch.device("cpu")
    print('GPU Not available')
print('--------------------------------')

for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=param.batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=len(dataset), sampler=test_subsampler)

    
    # Define model
    rad_model = DeepRadiomicsModel(param).to(device)
    
    rad_optimizer = optim.SGD(rad_model.parameters(), lr=param.lr, weight_decay=1e-6, momentum=0.9)

    #Set stoping condition
    min_loss = np.inf
    tolerance = 150
    count_tol = 0

    # Train the model
    for epoch in range(param.epochs):
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0
        loss_sum = 0
        y_list = []
        y_pred_list = []

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            
            # Get inputs
            x_radiomics, x_frames, gene_exp, labels = data
            x_radiomics, x_frames, gene_exp = x_radiomics.float().to(device), x_frames.float().to(device), gene_exp.float().to(device)

            # Zero the gradients
            rad_optimizer.zero_grad()
            
            # Perform forward pass
            exp_pred = rad_model(x_radiomics, x_frames)
            
            # Compute loss
            e_loss = exp_loss(exp_pred, gene_exp)
            
            # Perform backward pass
            e_loss.backward()
            
            # Perform optimization
            rad_optimizer.step()
            
            # Print statistics
            current_loss += e_loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

            loss_sum += e_loss.item()
        
        # add metrics to dataframe
        train_metrics = train_metrics.append({"epoch": epoch+1, "fold": fold, "loss": loss_sum}, ignore_index=True)     
        # Process is complete.
        print('Training process has finished. Saving trained model.')
        print(f'Train loss: {loss_sum:.3f}')

        # Print about testing
        print('Starting testing')

        # Evaluationfor this fold
        loss_sum = 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                x_radiomics, x_frames, gene_exp, labels = data
                x_radiomics, x_frames, gene_exp = x_radiomics.float().to(device), x_frames.float().to(device), gene_exp.float().to(device)

                # Perform forward pass
                exp_pred = rad_model(x_radiomics, x_frames)
                
                # Compute loss
                e_loss = exp_loss(exp_pred, gene_exp)


        # Print accuracy
        print('Loss for fold %d: %d' % (fold, e_loss))
        print('--------------------------------')

        if e_loss < min_loss:
            print('Loss decreased from %0.4f to %0.4f. Saving model...' % (min_loss, e_loss))
            print('--------------------------------')
            min_loss = e_loss
            results[fold] = min_loss
            count_tol = 0
            # Saving the model
            if not os.path.exists(os.path.join(param.checkpoint_dir, param.exp_id)):
                os.makedirs(os.path.join(param.checkpoint_dir, param.exp_id))
            torch.save(rad_model.state_dict(), os.path.join(param.checkpoint_dir, param.exp_id, 'expression-model-fold-{}.pth'.format(fold)))

        else: count_tol += 1
        if count_tol > tolerance: 
            print(f'Best performing epoch: {epoch-tolerance+1}')
            print('--------------------------------')
            break

        # add metrics to dataframe
        test_metrics = test_metrics.append({"epoch": epoch+1, "fold": fold, "loss": e_loss.item()}, ignore_index=True)
        
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {param.n_splits} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value}')
    sum += value
print(f'Average: {sum/len(results.items())}')

train_metrics.to_csv(os.path.join(param.checkpoint_dir, param.exp_id, 'train_metrics.csv'))
test_metrics.to_csv(os.path.join(param.checkpoint_dir, param.exp_id, 'test_metrics.csv'))
