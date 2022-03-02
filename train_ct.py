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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from utils.models import DirectPredictionCT
from utils.dataset import Dataset

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, default='./data')
argparser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
argparser.add_argument('--exp_id', type=str, default='./ig_100')
argparser.add_argument('--exp_model', type=str, default='resnet18')
argparser.add_argument('--lr', type=float, default=0.00001)
argparser.add_argument('--one_fold', action='store_true')
argparser.add_argument('--fold', type=int, default=4)
argparser.add_argument('--n_splits', type=int, default=5)
argparser.add_argument('--seed', type=int, default=42)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--epochs', type=int, default=10000)
argparser.add_argument('--image_channels', type=int, default=3, help='options: 3, 9')
argparser.add_argument('--image_size', type=int, default=224, help='options: 32, 64, 112, 224')
argparser.add_argument('--load_manual_features', action='store_true')
argparser.add_argument('--transforms', type=str, default=None)
param = argparser.parse_args()

train_metrics = pd.DataFrame(columns=["epoch", "fold", "loss", "accuracy", "roc_auc", "precision", "recall", "f1"])
test_metrics = pd.DataFrame(columns=["epoch", "fold", "loss", "accuracy", "roc_auc", "precision", "recall", "f1"])

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

# set criterion as loss for regression
weights = torch.FloatTensor([0.2778, 0.7222]).to(device)
exp_loss = nn.CrossEntropyLoss(weight=weights)
results = {}

y = dataset.labels
x = dataset.labels.index

# set up cross validation
# kf = KFold(n_splits=param.n_splits, shuffle=True, random_state=param.seed)
skf = StratifiedKFold(n_splits=param.n_splits, shuffle=True, random_state=param.seed)

for fold, (train_ids, test_ids) in enumerate(skf.split(x, y)):

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
                        batch_size=len(test_ids), sampler=test_subsampler)

    # Save input
    input_dir = os.path.join(param.checkpoint_dir, param.exp_id, 'input', 'fold_{}'.format(fold))
    if not os.path.exists(input_dir):
                os.makedirs(input_dir)
    for i, batch in enumerate(trainloader):
        torchvision.utils.save_image(batch[0][:,0:1,:,:], os.path.join(input_dir, 'train_{}.png'.format(i)), normalize=True)
    for i, batch in enumerate(testloader):
        torchvision.utils.save_image(batch[0][:,0:1,:,:], os.path.join(input_dir, 'test_{}.png'.format(i)), normalize=True)

    # Define model
    ct_model = DirectPredictionCT(param).to(device)
    ct_optimizer = optim.AdamW(ct_model.parameters(), lr=param.lr)

    #Set stoping condition
    min_acu = 0
    min_loss = np.inf
    tolerance = 50
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

        ct_model.train()
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            
            # Get inputs
            x_frames, labels = data
            x_frames, labels = x_frames.float().to(device), labels.to(device)
            labels = labels.squeeze(dim=1)

            # Zero the gradients
            ct_optimizer.zero_grad()
            
            # Perform forward pass
            r_pred = ct_model(x_frames)
            
            # Compute loss
            loss = exp_loss(r_pred, labels)
                       
            # Perform backward pass
            loss.backward()

            # Perform optimization
            ct_optimizer.step()
            
            # Print statistics
            current_loss += loss.item()

            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

            loss_sum += loss.item()
            y_list.extend(labels.view(-1).cpu().numpy())
            y_pred_list.extend(torch.tensor([torch.argmax(a) for a in r_pred]).numpy())

        train_acc = accuracy_score(y_list, y_pred_list)
        train_roc_auc = roc_auc_score(y_list, y_pred_list)
        train_precision = precision_score(y_list, y_pred_list)
        train_recall = recall_score(y_list, y_pred_list)
        train_f1 = f1_score(y_list, y_pred_list)

        if train_acc > min_acu:
            print('Accuracy increased from %0.4f to %0.4f. Saving model...' % (min_acu, train_acc))
            print('--------------------------------')
            min_acu = train_acc
            count_tol = 0
            # Saving the model
            torch.save(ct_model.state_dict(), os.path.join(param.checkpoint_dir, param.exp_id, 'CT-model-fold-{}.pth'.format(fold)))

        # if loss_sum < min_loss:
        #     print('Loss decreased from %0.4f to %0.4f. Saving model...' % (min_loss, loss_sum))
        #     print('--------------------------------')
        #     min_loss = loss_sum
        #     count_tol = 0
        #     # Saving the model
        #     torch.save(ct_model.state_dict(), os.path.join(param.checkpoint_dir, param.exp_id, 'CT-model-fold-{}.pth'.format(fold)))

        else: count_tol += 1
        
        # add metrics to dataframe
        train_metrics = train_metrics.append({"epoch": epoch+1, "fold": fold, "loss": loss_sum, "accuracy": train_acc, "roc_auc": train_roc_auc, "precision": train_precision, "recall": train_recall, "f1": train_f1}, ignore_index=True)     
        # Process is complete.
        print('Training process has finished.')
        print(f'Train Loss: {loss_sum:.3f}')
        print(f'Train Accuracy: {train_acc:.3f}')
        print(f'Train ROC AUC: {train_roc_auc:.3f}')
        print(f'Train Precision: {train_precision:.3f}')
        print(f'Train Recall: {train_recall:.3f}')
        print(f'Train F1: {train_f1:.3f}')
        print('--------------------------------')

        # Print about testing
        print('Starting testing')

        # Evaluationfor this fold
        correct, total = 0, 0
        test_loss = 0.0
        y_list = []
        y_pred_list = []
        ct_model.eval()

        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                x_frames, labels = data
                x_frames, labels = x_frames.float().to(device), labels.to(device)
                labels = labels.squeeze(dim=1)

                # Perform forward pass
                r_pred = ct_model(x_frames)
                
                # Compute loss
                loss = exp_loss(r_pred, labels)

                # # Set total and correct
                # _, predicted = torch.max(r_pred.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                test_loss += loss.item()
                y_list.extend(labels.view(-1).cpu().numpy())
                y_pred_list.extend(torch.tensor([torch.argmax(a) for a in r_pred]).numpy())

        # Print accuracy
        print('Test metrics for fold %d:' % (fold))
        test_acc = accuracy_score(y_list, y_pred_list)
        test_roc_auc = roc_auc_score(y_list, y_pred_list)
        test_precision = precision_score(y_list, y_pred_list)
        test_recall = recall_score(y_list, y_pred_list)
        test_f1 = f1_score(y_list, y_pred_list)
        results[fold] = test_acc

        print(f'Test Loss: {test_acc:.3f}')
        print(f'Test Accuracy: {test_acc:.3f}')
        print(f'Test ROC AUC: {test_roc_auc:.3f}')
        print(f'Test Precision: {test_precision:.3f}')
        print(f'Test Recall: {test_recall:.3f}')
        print(f'Test F1: {test_f1:.3f}')
        print('--------------------------------')

        # add metrics to dataframe
        test_metrics = test_metrics.append({"epoch": epoch+1, "fold": fold, "loss": test_loss, "accuracy": test_acc, "roc_auc": test_roc_auc, "precision": test_precision, "recall": test_recall, "f1": test_f1}, ignore_index=True)
        
        if count_tol > tolerance: 
            print(f'Best performing epoch: {epoch-tolerance+1}')
            print('--------------------------------')
            break

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
