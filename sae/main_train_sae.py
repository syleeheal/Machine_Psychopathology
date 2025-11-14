import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import random
import torch
import torch.optim as optim
import torch.nn as nn
import wandb

from utils import Config, Data_Manager, load_label_dict, data_split
from sae import S3AE, train_sae, infer_sae
from utils_sae_eval import eval_sae_stat, eval_sae_cls, eval_sae_cls_thought, eval_sae_activation_cls, eval_sae_recon, eval_sae_direction

import argparse
import tabulate

import pandas as pd
import itertools
import re
import pickle

def parameter_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0", )
    parser.add_argument("--epochs", type=int, default=900, )
    parser.add_argument("--lr", type=float, default=0.001, )
    parser.add_argument("--dr", type=float, default=0.0,) 
    parser.add_argument("--grad-norm", type=float, default=5,) 
    parser.add_argument("--hid-dim-factor", type=float, default=2,)
    parser.add_argument("--batch-size", type=int, default=8192,)
    parser.add_argument("--hook-layer", type=int, default=11,)
    parser.add_argument("--tgt-init-recon-loss", type=float, default=10,) # llama 10 # gemma 500
    parser.add_argument("--tgt-init-cls-loss", type=float, default=5,) # llama 5 # gemma 5
    parser.add_argument("--wandb", action='store_true', default=False)

    return parser.parse_args()


def set_config(args):
    """
    CONFIG
    """
    cfg = Config()
    cfg.hook_layers = [args.hook_layer]
    dm = Data_Manager(cfg)

    X_dict, y = dm.load_dict(dict_type='activation')
    std_dict = dm.load_dict(dict_type='act-std')
    
    X = X_dict[args.hook_layer]
    X = X / std_dict[args.hook_layer]  # standardize by std

    print(f"X: {X.shape}, y: {y.shape}")
    
    symp_label_dict, dim_dict, sub_dim_dict, symp_keys, dim_keys, subdim_keys, belief_keys = dm.load_dict(dict_type='label') # type: ignore

    assert len(symp_label_dict) == y.shape[1], f"Label dict size {len(symp_label_dict)} does not match y shape {y.shape[1]}"
    
    return cfg, X, y, symp_label_dict


def model_train(args, cfg, X, y):
    """
    TRAIN SAE
    """
    
    # loss weights
    label_weight = y.sum() / y.sum(0)
    label_weight = ((label_weight / label_weight.sum()) * y.shape[1])


    initial_mse_loss = (X**2).mean()
    recon_weight = args.tgt_init_recon_loss / initial_mse_loss


    initial_cls_loss = nn.BCEWithLogitsLoss(weight=label_weight)(torch.zeros_like(y), y)
    cls_weight = args.tgt_init_cls_loss / initial_cls_loss

    label_weight = label_weight.to(args.device)
    recon_weight = recon_weight.to(args.device);
    cls_weight = cls_weight.to(args.device)

    # wandb run
    run = None
    model_id = re.sub(r'/', '-', cfg.model_id)
    if args.wandb:
        run = wandb.init(
            project=f"{model_id}{cfg.feat_type}-{args.hook_layer}-v{cfg.current_v}", 
            config={"architecture": f"Supervised AE-v{cfg.current_v}",}
        )

    # setup data and model
    dataloader_train, dataloader_test = data_split(X, y, args.batch_size, train_size=1)

    sae = S3AE(
        input_dim=X.shape[1], 
        hidden_dim=int(X.shape[1] * args.hid_dim_factor),
        label_dim=y.shape[1],
    ).to(X.dtype)

    sae.to(args.device)
    
    # train
    optimizer = optim.Adam(sae.parameters(), lr=0.01, weight_decay=0)
    sae, optimizer, best_loss = train_sae(
        sae, optimizer, dataloader_train, dataloader_test, 
        num_epochs=args.epochs,
        run=run,
        grad_norm=args.grad_norm, 
        label_weight=label_weight, 
        recon_weight=recon_weight, 
        cls_weight=cls_weight
    )

    # close wandb run
    if run is not None:
        run.finish()

    return sae, dataloader_test, optimizer, best_loss


def model_eval(sae, dataloader_test, symp_label_dict):
    """
    EVAL SAE
    """
    X_test, X_hat_test, Z_test, y_test, y_hat_test = infer_sae(sae, dataloader_test)

    eval_sae_stat(X_test, X_hat_test, Z_test)
    eval_sae_cls(y_test, y_hat_test)
    for thought in symp_label_dict.keys():
        print(f"\033[1m\033[92m{thought}\033[0m")
        f1 = eval_sae_cls_thought(thought, y_test, y_hat_test, symp_label_dict)
        sample_ratio, change_percent = eval_sae_recon(thought, sae, X_test, y_test, symp_label_dict)
        sensitivity, specificity = eval_sae_activation_cls(thought, Z_test, y_test, symp_label_dict, activation_threhold=1e-3)
        print(f"F1 Score: {f1:.2f}")
        print(f"Sample Ratio: {sample_ratio:.3f}")
        print(f"Change Percent: {change_percent:.3f}")
        print()


if __name__ == "__main__":

    args = parameter_parser()

    lrs = [0.005, 0.002, 0.001]
    grad_norms = [0.5, 0.2, 0.1]
    
    hyperparams = list(itertools.product(lrs, grad_norms))
    best_loss_per_hyperparam = 1e10
    
    for lr, grad_norm in hyperparams:
        args.lr = lr
        args.grad_norm = grad_norm
    
        print(tabulate.tabulate(vars(args).items(), tablefmt='fancy_grid'))

        cfg, X, y, symp_label_dict = set_config(args)
        sae, dataloader_test, optimizer, best_loss = model_train(args, cfg, X, y)

        if best_loss < best_loss_per_hyperparam:
            best_loss_per_hyperparam = best_loss
            best_sae = sae
            sae_path = f'./{cfg.sae_dir}/layer_{args.hook_layer}/sae{cfg.feat_type}_{args.hook_layer}_v{cfg.current_v}.pt'
            torch.save(best_sae.eval().cpu().state_dict(), sae_path)
            best_hyperparams = (args.lr, args.grad_norm)
        
    model_eval(best_sae.to(args.device), dataloader_test, symp_label_dict)

    
    X_test, X_hat_test, Z_test, Y_test, Y_hat_test = infer_sae(best_sae.to(args.device), dataloader_test)
    act_max = Z_test[:,:len(symp_label_dict)].max(0)[0].tolist()
    act_max_dict = dict(zip(symp_label_dict.keys(), act_max))
    actmax_path = f'{cfg.sae_dir}/layer_{args.hook_layer}/actmax{cfg.feat_type}_{args.hook_layer}_v{cfg.current_v}.pkl'
    with open(actmax_path, 'wb') as f:
        pickle.dump(act_max_dict, f)

    print(f"Best hyperparameters: lr={best_hyperparams[0]}, grad_norm={best_hyperparams[1]}")
    with open(f'{cfg.sae_dir}/layer_{args.hook_layer}/best_hyperparams{cfg.feat_type}_{args.hook_layer}_v{cfg.current_v}.txt', 'w') as f:
        f.write(f"Best hyperparameters: lr={best_hyperparams[0]}, grad_norm={best_hyperparams[1]}\n")
        f.write(f"Best loss: {best_loss_per_hyperparam}\n")
        f.write(f"args: {vars(args)}\n")
