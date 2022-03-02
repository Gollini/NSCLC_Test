from cProfile import label
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SelectFpr, SelectKBest, chi2, f_classif, mutual_info_classif
from tabgan.sampler import OriginalGenerator, GANGenerator
from pytorch_lightning import Trainer, seed_everything
from utils.models import GeneAE
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/gene_data/drop_nan/117")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--recurrence_time_period", type=str, default="1y", help="options: 6m, 1y, 18m, 2y, 30m, 4y, 5y")
    parser.add_argument("--feature_selection", type=str, default="none", help="options: none, f_test, chi2, mutual_info, all")
    parser.add_argument("--feature_selection_alpha", type=float, default=0.02)
    parser.add_argument("--data_scaler", type=str, default="standard", help="options: none, standard, minmax, log, log+1")
    parser.add_argument("--data_aug_method", type=str, default="none", help="options: none, original, gan, gaussian")
    parser.add_argument("--data_aug_times", type=int, default=1, help="if data_aug_method is not none, this is the number of times to augment the data")
    parser.add_argument("--downstream_epochs", type=int, default=1000)
    parser.add_argument("--pretrain_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=93)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one_fold", action="store_true")
    parser.add_argument("--fold_idx", type=int, default=0, help="if one_fold is set to True, this is the fold_idx number")
    parser.add_argument("--k", type=int, default=5, help="number of folds for k-fold cross validation if one_fold is set to False")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--pretraining_patience", type=int, default=10)
    parser.add_argument("--downstream_patience", type=int, default=10)
    parser.add_argument("--class_0_weight", type=float, default=0.5, help="weight of class 0 in the loss function")

    parser = GeneAE.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    param = parser.parse_args()
    return param

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)

def load_data(param):
    labels = pd.read_csv(os.path.join(param.data_dir, 'labels_time.csv'), index_col=0)
    labels = labels.replace(to_replace=['no'], value=0)
    labels = labels.replace(to_replace=['yes'], value=1)
    if param.recurrence_time_period == "6m":
        labels = labels[labels['Recurrence within 6 months'].isna() == False]['Recurrence within 6 months']
    elif param.recurrence_time_period == "1y":
        labels = labels[labels['Recurrence within 1 year'].isna() == False]['Recurrence within 1 year']
    elif param.recurrence_time_period == "18m":
        labels = labels[labels['Recurrence within 18 months'].isna() == False]['Recurrence within 18 months']
    elif param.recurrence_time_period == "2y":
        labels = labels[labels['Recurrence within 2 years'].isna() == False]['Recurrence within 2 years']
    elif param.recurrence_time_period == "30m":
        labels = labels[labels['Recurrence within 30 months'].isna() == False]['Recurrence within 30 months']
    elif param.recurrence_time_period == "4y":
        labels = labels[labels['Recurrence within 4 years'].isna() == False]['Recurrence within 4 years']
    elif param.recurrence_time_period == "5y":
        labels = labels[labels['Recurrence within 5 years'].isna() == False]['Recurrence within 5 years']
    df = pd.read_csv(os.path.join(param.data_dir, 'gene_exp.csv'), index_col=0).loc[labels.index,:]
    labels = labels.astype(int)
    return df, labels

def scale_data(X_train, X_test, param):
    if param.data_scaler == "standard":
        scaler = StandardScaler()
    elif param.data_scaler == "minmax":
        scaler = MinMaxScaler()
    elif param.data_scaler == "log":
        scaler = FunctionTransformer(np.log1p, validate=True)
    elif param.data_scaler == "log+1":
        scaler = FunctionTransformer(lambda x: np.log1p(x + 1), validate=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_train, X_test

def select_features(X_train, X_test, y_train, param):
    if param.feature_selection == "f_test":
        selector = SelectFpr(f_classif, alpha=param.feature_selection_alpha)
    elif param.feature_selection == "chi2":
        selector = SelectFpr(chi2, alpha=param.feature_selection_alpha)
    elif param.feature_selection == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=125)
    elif param.feature_selection == "all":
        ft = SelectFpr(f_classif, alpha=param.feature_selection_alpha)
        X_train_1 = ft.fit_transform(X_train, y_train)
        X_test_1 = ft.transform(X_test)
        chi = SelectFpr(chi2, alpha=param.feature_selection_alpha)
        X_train_2 = chi.fit_transform(X_train, y_train)
        X_test_2 = chi.transform(X_test)
        mi = SelectKBest(mutual_info_classif, k=125)
        X_train_3 = mi.fit_transform(X_train, y_train)
        X_test_3 = mi.transform(X_test)

        # take intersection of all three sets
        ft_mask = ft.get_support()
        chi_mask = chi.get_support()
        mi_mask = mi.get_support()
        # select indices of ft_mask that are true
        ft_indices = [i for i, x in enumerate(ft_mask) if x]
        # select indices of chi_mask that are true
        chi_indices = [i for i, x in enumerate(chi_mask) if x]
        # select indices of mi_mask that are true
        mi_indices = [i for i, x in enumerate(mi_mask) if x]
        # take intersection of all three sets
        indices = list(set(ft_indices) & set(chi_indices) & set(mi_indices))
        X_train = X_train[:, indices]
        X_test = X_test[:, indices]

    if param.feature_selection != "all":
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    train_n_features = X_train.shape[1]
    test_n_features = X_test.shape[1]

    return X_train, X_test, train_n_features, test_n_features


def augment_data(X_train, X_test, y_train, param):
    if param.data_aug_method == "gaussian":
        # fit a gaussian distribution to each feature
        # and then sample from it to generate data
        # augmentation
        X_aug, y_aug = [], []
        # count number of unique classes in y_train
        unique_classes = np.unique(y_train)
        # for each class
        for c in unique_classes:
            # get indices of rows in X_train that have class i
            indices = np.where(y_train == c)[0]
            X_train_aug = np.zeros((len(indices) * param.data_aug_times, X_train.shape[1]))
            y_train_aug = np.ones(len(indices) * param.data_aug_times) * c
            for j in range(X_train.shape[1]):
                X_train_aug[:,j] = np.random.normal(loc=X_train[indices,j].mean(), scale=X_train[indices,j].std(), size=len(indices) * param.data_aug_times)
            X_aug.append(X_train_aug)
            y_aug.append(y_train_aug)
        X_train = np.concatenate((X_train, np.concatenate(X_aug)), axis=0)
        y_train = np.concatenate((y_train, np.expand_dims(np.concatenate(y_aug), 1)), axis=0)
        
    else:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        if param.data_aug_method == "original":
            X_train, y_train = OriginalGenerator(gen_x_times=param.data_aug_times, is_post_process=False).generate_data_pipe(X_train, y_train, X_test, )
        
        elif param.data_aug_method == "gan":
            X_train, y_train = GANGenerator(gen_x_times=param.data_aug_times, is_post_process=False).generate_data_pipe(X_train, y_train, X_test, )

        X_train, y_train = X_train.values, y_train.values.reshape(-1,1)
        y_train[y_train < 0] = 0

    return X_train, y_train


def train_val_split(df, labels, train_index, test_index):
    X_train, X_test = df.iloc[train_index,:].values, df.iloc[test_index,:].values
    y_train, y_test = labels.iloc[train_index].values, labels.iloc[test_index].values

    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, X_test, y_train, y_test, param):
    trainset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    testset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(trainset, batch_size=param.batch_size, drop_last=True)
    test_loader = DataLoader(testset, batch_size=X_test.shape[0], drop_last=True)
    
    return train_loader, test_loader

def calculate_class_weights(y_train):
    class_counts = np.bincount(y_train)
    class_weights = class_counts / class_counts.sum()
    return np.flip(class_weights)

def define_callbacks_loggers_pretraining(param, count):
    checkpoint_path = os.path.join(param.checkpoint_dir, param.exp_name, 'fold-{}'.format(count))
    tb_logger = pl_loggers.TensorBoardLogger(checkpoint_path, name='pretraining')
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='pretraining')
    early_stopping = EarlyStopping('val_recon_loss', patience=param.pretraining_patience)
    model_checkpoint = ModelCheckpoint(tb_logger.log_dir, monitor='val_recon_loss', mode='min', save_top_k=1)
    wandb_logger = WandbLogger(project = 'nsclc_rec_gene', group = '{}-pretraining'.format(param.exp_name), name = 'fold-{}'.format(count))
    return early_stopping, model_checkpoint, tb_logger, wandb_logger, csv_logger, checkpoint_path

def define_callbacks_loggers_downstream(param, checkpoint_path, count):
    tb_logger = pl_loggers.TensorBoardLogger(checkpoint_path, name='downstream')
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='downstream')
    early_stopping = EarlyStopping('val_down_loss', patience=param.downstream_patience)
    model_checkpoint = ModelCheckpoint(tb_logger.log_dir, monitor='val_accuracy', mode='max', save_top_k=1)
    wandb_logger = WandbLogger(project = 'nsclc_rec_gene', group = '{}-downstream'.format(param.exp_name), name = 'fold-{}'.format(count))
    return early_stopping, model_checkpoint, tb_logger, wandb_logger, csv_logger

def calculate_metrics(param, csv_logger, tb_logger):
    val_metrics = []
    train_metrics = []
    for count in range(param.k):
        checkpoint_path = os.path.join(param.checkpoint_dir, param.exp_name, 'fold-{}'.format(count), 'downstream', 'version_')
        # get the file that starts with epoch
        epoch_files = [f for f in os.listdir(checkpoint_path + str(tb_logger.version)) if f.startswith('epoch')]
        best_epoch = int(epoch_files[0][6])
        # read metrics.csv in checkpoint_path
        metrics_df = pd.read_csv(os.path.join(checkpoint_path + str(csv_logger.version), 'metrics.csv'))
        best_metrics = metrics_df[metrics_df['epoch']==best_epoch]
        # choose valus that are not nan from best_metrics
        val_metrics.append(best_metrics.iloc[0,:8])
        train_metrics.append(best_metrics.iloc[1,[7,9,10,11]])
    val_metrics = pd.DataFrame(val_metrics)
    val_metrics.loc['mean'] = val_metrics.mean()
    val_metrics.to_csv(os.path.join(param.checkpoint_dir, param.exp_name, 'avg_val_metrics.csv'))
    train_metrics = pd.DataFrame(train_metrics)
    train_metrics.loc['mean'] = train_metrics.mean()
    train_metrics.to_csv(os.path.join(param.checkpoint_dir, param.exp_name, 'avg_train_metrics.csv'))