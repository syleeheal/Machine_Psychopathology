import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import gc
import scipy.stats as statsc
import matplotlib.pyplot as plt
import re


def eval_sae_stat(X, X_hat, Z):
    """
    measure statistics of the SAE
    """
    
    scale = torch.abs(X).mean().item()
    l1 = torch.abs(X-X_hat).mean().item()
    act_Z = ((abs(Z) > 1e-3).sum(dim=0) > 0).sum().item()
    scale_Z = Z[abs(Z) > 1e-3].mean().item(), Z[abs(Z) > 1e-3].std().item()
    dense_Z = (abs(Z) > 1e-3).float().mean().item()

    print('Scale of X:', scale)
    print('L1 distance to X_hat:', l1)
    print('Num active Z:', act_Z)
    print('Scale of active Z:', scale_Z)
    print('Density of Z:', dense_Z)
    print()
    
def eval_sae_cls(Y, Y_hat):
    """
    measure overall F1
    """
    _Y_hat = Y_hat.sigmoid()
    _Y_hat = (_Y_hat > 0.5).float()
    
    f1_macro = f1_score(Y.cpu().numpy(), _Y_hat.cpu().numpy(), average='macro')
    f1_micro = f1_score(Y.cpu().numpy(), _Y_hat.cpu().numpy(), average='micro')
    print(f'F1 macro: {f1_macro}, F1 micro: {f1_micro}', '\n')    

def eval_sae_cls_thought(thought, Y, Y_hat, label_dict):
    """
    measure F1 for the thought
    """
    thought_idx = label_dict[thought]
    
    Y = Y[:, thought_idx]
    Y_hat = Y_hat[:, thought_idx]
    
    Y_hat = Y_hat.sigmoid()
    Y_hat = (Y_hat > 0.5).float()

    
    f1 = f1_score(Y.cpu().numpy(), Y_hat.cpu().numpy(), average='binary')
    return f1

def eval_sae_activation_cls(thought, Z, Y, label_dict, activation_threhold=1e-3):
    """
    measure sensitivity and specificity for each thought
    """

    thought_idx = label_dict[thought]
    
    thought_act = Z[:,thought_idx]
    thought_act = (thought_act > activation_threhold).float()

    true_positive = ((thought_act == 1) & (Y[:,thought_idx] == 1)).float().sum().item()
    false_positive = ((thought_act == 1) & (Y[:,thought_idx] == 0)).float().sum().item()
    true_negative = ((thought_act == 0) & (Y[:,thought_idx] == 0)).float().sum().item()
    false_negative = ((thought_act == 0) & (Y[:,thought_idx] == 1)).float().sum().item()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    return sensitivity, specificity

def eval_sae_recon(thought, sae, X, Y, label_dict):
    """
    measure loss increase when deactivating the thought
    """
    
    thought_idx = label_dict[thought]
    device = sae.encoder.weight.device
    
    X_t = X[Y[:, thought_idx] == 1].to(device)
    
    mask = torch.ones_like(sae.decoder.weight.T)
    mask[thought_idx] = 0
    
    batch_size = 8192
    
    X_hat_ts = []
    losses = []
    X_hat_t_deacts = []
    losses_deact = []
    
    for i in range(0, X_t.size(0), batch_size):
        X_t_batch = X_t[i:i+batch_size]
        
        X_hat_t = ((X_t_batch @ sae.encoder.weight.T).relu() @ (sae.decoder.weight.T))
        loss = torch.nn.functional.mse_loss(X_hat_t, X_t_batch, reduction='none').mean(dim=1)
        
        X_hat_t_deact = ((X_t_batch @ sae.encoder.weight.T).relu() @ (sae.decoder.weight.T * mask))
        loss_deact = torch.nn.functional.mse_loss(X_hat_t_deact, X_t_batch, reduction='none').mean(dim=1)

        X_hat_t = X_hat_t.detach().cpu()
        loss = loss.detach().cpu()
        X_hat_t_deact = X_hat_t_deact.detach().cpu()
        loss_deact = loss_deact.detach().cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        X_hat_ts.append(X_hat_t)
        losses.append(loss)
        X_hat_t_deacts.append(X_hat_t_deact)
        losses_deact.append(loss_deact)
        
    
    X_hat_t = torch.cat(X_hat_ts, dim=0)
    loss = torch.cat(losses, dim=0)
    X_hat_t_deact = torch.cat(X_hat_t_deacts, dim=0)
    loss_deact = torch.cat(losses_deact, dim=0)

    diff = (loss_deact - loss) 
    sample_ratio = ((diff > 0).sum() / diff.size(0))
    change_percent = (loss_deact.mean() / loss.mean()).item()
    
    return sample_ratio.item(), change_percent

def eval_sae_direction(thought, sae, label_dict):
    """
    measure cosine similarity between feature directions
    """
    
    thought_idx = label_dict[thought]
    num_thoughts = len(label_dict)
    _label_dict = {v: k for k, v in label_dict.items()}
    
    feat_dirs = sae.decoder.weight.T[:num_thoughts].detach().cpu().float()
    cos_sim = torch.nn.functional.cosine_similarity(feat_dirs.unsqueeze(1)[thought_idx], feat_dirs.unsqueeze(0), dim=-1)
    
    thought_idx_list = [_label_dict[i] for i in range(len(_label_dict))]

    # visualize cos_sim as heatmap
    plt.figure(figsize=(20, 1))
    sns.heatmap(cos_sim, annot=True, cmap='Blues', xticklabels=thought_idx_list)
    plt.title('Cosine Similarity of Feature Directions')
    plt.show()

def eval_sae_activation_corr(cfg, df, measure_manager, symp_keys, label_dict, sev_dict, layers):

    df['sev_pred'] = np.nan
    dfs = []
    batch_size = 256

    for t in symp_keys:

        _df = df[df['label'] == t]
        if len(_df) == 0: continue
        
        prompts = _df['output_text'].tolist()
        
        sae_preds = []
        for j in range(0, len(prompts), batch_size):
            sae_pred = measure_manager.sae_measure_no_json(prompts[j:j+batch_size])
            sae_preds.extend(sae_pred)
        
        _df['sev_pred'] = sae_preds
        dfs.append(_df)

        torch.cuda.empty_cache()
        gc.collect()

    df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return df
