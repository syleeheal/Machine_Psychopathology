import torch
import torch.optim as optim
import wandb

from s3ae import *
from s3ae_eval import *

import argparse
import tabulate

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from utils import get_dicts

def parameter_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0", )
    parser.add_argument("--epochs", type=int, default=1000, )
    parser.add_argument("--lr", type=float, default=0.001, )
    parser.add_argument("--dr", type=float, default=0.0,) 
    parser.add_argument("--grad-norm", type=float, default=5,) 
    parser.add_argument("--hid-dim-factor", type=int, default=1,)
    parser.add_argument("--batch-size", type=int, default=8192,)
    parser.add_argument("--path-X", type=str, default='./data/X.pt', )
    parser.add_argument("--path-y", type=str, default='./data/y.pt', )
    parser.add_argument("--path-s3ae", type=str, default='./model/trained_s3ae.pt', )

    return parser.parse_args()


def set_config():
    """
    CONFIG
    """    

    label_dict, act_max_dict, act_labels, abbv_dict = get_dicts()

    X = torch.load(f'{args.path_X}', weights_only=True)
    y = torch.load(f'{args.path_y}', weights_only=True)
    
    # sample 1/10 of X and y
    idx = torch.randperm(X.size(0))[:int(X.size(0)/10)]
    X = X[idx]
    y = y[idx]
    
    print(f"X: {X.shape}, y: {y.shape}")
    
    return X, y, label_dict


def data_split(X, y, batch_size, train_size=1):
    
    if train_size != 1:
        train_idx = torch.randperm(X.size(0))[:int(train_size*X.size(0))]
        test_idx = torch.randperm(X.size(0))[int(train_size*X.size(0)):]
    else:
        train_idx = torch.arange(X.size(0))
        test_idx = torch.arange(X.size(0))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        
    return dataloader_train, dataloader_test


def model_train(args, X, y):
    """
    TRAIN SAE
    """
        
    cls_weight = y.sum() / y.sum(0)
    cls_weight = ((cls_weight / cls_weight.sum()) * y.shape[1]).to(args.device)    

    dataloader_train, dataloader_test = data_split(X, y, args.batch_size, train_size=1)

    sae = S3AE(
        input_dim=X.shape[1], 
        hidden_dim=int(X.shape[1] * args.hid_dim_factor),
        label_dim=y.shape[1],
    ).to(X.dtype)
    sae.to(args.device)
    
    optimizer = optim.Adam(sae.parameters(), lr=args.lr, weight_decay=args.dr)

    sae, optimizer = train(
        args, sae, optimizer, dataloader_train,
        num_epochs=args.epochs,
        grad_norm=args.grad_norm,
        class_loss=True,
        cls_weight=cls_weight,
    )
    
    return sae, dataloader_test, optimizer


def model_eval(sae, dataloader_test, label_dict):
    """
    EVAL SAE
    """
    X_test, X_hat_test, Z_test, Y_test, Y_hat_test = infer_sae(sae.to(args.device), dataloader_test)

    act_max = Z_test[:,:len(label_dict)].max(0)[0].tolist()
    act_max_dict = dict(zip(label_dict.keys(), act_max))
    print(act_max_dict)

    eval_sae_stat(X_test, X_hat_test, Z_test)
    eval_sae_cls(Y_test, Y_hat_test)
    for thought in label_dict.keys():
        print(f"\033[1m\033[92m{thought}\033[0m")
        eval_sae_cls_thought(thought, Y_test, Y_hat_test, label_dict)
        eval_sae_recon(thought, sae, X_test, Y_test, label_dict)
        eval_sae_activation_cls(thought, Z_test, Y_test, label_dict, activation_threhold=1e-3)
        eval_sae_direction(thought, sae, label_dict)
        print()        
        

def train(args, sae, optimizer, dataloader_train, num_epochs, grad_norm=5, class_loss=True, cls_weight=None):
    
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.BCEWithLogitsLoss(weight=cls_weight)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30,T_mult=1,eta_min=1e-5,)
    
    iterator = trange(num_epochs, desc='Cls. Loss: ', leave=False)
    best_loss = float('inf')
    patience = 0
    
    for epoch in iterator:
        
        train_epoch_loss = 0
        for X, Y in dataloader_train:
            
            X = X.to(args.device)
            Y = Y.to(args.device)
            
            output, z, y_pred = sae(X)

            l2_norm = sae.decoder.weight.norm(dim=0, p=2)
            sparsity_loss = (z*l2_norm).mean()
            recon_loss = criterion_1(output, X)
            class_loss = criterion_2(y_pred, Y)
            train_loss = (recon_loss) + (sparsity_loss) + (class_loss)
            train_epoch_loss += train_loss.item()
            
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), grad_norm)
            optimizer.step()
            
            iterator.set_description('Cls. Loss: {:.4f}'.format(class_loss.item()))
        
        patience += 1
        if train_epoch_loss < best_loss:
            best_loss = train_epoch_loss
            best_s3ae = sae
            patience = 0
        
        scheduler.step()
    
    print('Reconstruction Loss:', recon_loss.item(), 'Sparsity Loss:', sparsity_loss.item(), 'Class Loss:', class_loss.item())
    print('Training complete.')
        
    return best_s3ae, optimizer


if __name__ == "__main__":
    args = parameter_parser()
    print(tabulate.tabulate(vars(args).items(), tablefmt='fancy_grid'))
    X, y, label_dict = set_config()
    sae, dataloader_test, optimizer = model_train(args, X, y)
    model_eval(sae, dataloader_test, label_dict)
    torch.save(sae.eval().cpu().state_dict(), f'{args.path_s3ae}')
    
    
    