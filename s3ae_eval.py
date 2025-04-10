
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
    
    print('Scale of X:', torch.abs(X).mean().item())
    print('L1 distance to X_hat:', torch.abs(X-X_hat).mean().item())
    print('Num active Z:', ((Z > 1e-3).sum(dim=0) > 0).sum().item())
    print('Scale of active Z:', Z[Z > 1e-3].mean().item(), Z[Z > 1e-3].std().item())
    print('Density of Z:', (Z > 1e-3).float().mean().item())
    print('Ratio of dead Z:', (Z < 1e-3).all(dim=0).float().mean().item())
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
    print(f'F1 for {thought}: {f1}', '\n')

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

    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)

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
        
    # X_hat_t = ((X_t @ sae.encoder.weight.T).relu() @ (sae.decoder.weight.T))
    # loss = torch.nn.functional.mse_loss(X_hat_t, X_t, reduction='none').mean(dim=1)

    # X_hat_t_deact = ((X_t @ sae.encoder.weight.T).relu() @ (sae.decoder.weight.T * mask))
    # loss_deact = torch.nn.functional.mse_loss(X_hat_t_deact, X_t, reduction='none').mean(dim=1)
    
    X_hat_t = torch.cat(X_hat_ts, dim=0)
    loss = torch.cat(losses, dim=0)
    X_hat_t_deact = torch.cat(X_hat_t_deacts, dim=0)
    loss_deact = torch.cat(losses_deact, dim=0)

    diff = (loss_deact - loss) 
    sample_ratio = ((diff > 0).sum() / diff.size(0))
    change_percent = (loss_deact.mean() / loss.mean()).item()
    
    print(f'Sample ratio of loss increase: {sample_ratio.item():.4f}')
    print(f'Mean loss change: {change_percent*100:.1f}%')

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

def eval_sae_activation_corr(cfg, df, llm_model, sae, act_labels, label_dict, sev_dict, act_max_dict):

    batch_size = 200
    for t in act_labels:

        _df = df[df['label'] == t]
        if len(_df) == 0: continue
        
        prompts = _df['output_text'].tolist()
        
        # for each prompt, remove special character if it appears at the beginning or the end of the prompt
        for i, prompt in enumerate(prompts):
            prompts[i] = re.sub(r'^[^\w]+', '', prompt)
            prompts[i] = re.sub(r'[^\w]+$', '', prompts[i])
            
        outs = []
        for j in range(0, len(prompts), batch_size):
            prompt_batch = prompts[j:j+batch_size]
            with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                out = llm_model.run_with_cache(prompt_batch)
                out = out[1][cfg.hook_name].mean(dim=1)
                outs.append(out)
        out = torch.cat(outs, dim=0)
        
        X_hat, Z, Y_hat = sae(out)
        sae_preds = Z[:, :len(label_dict)]
        sev_preds = sae_preds[:, label_dict[t]].float().detach().cpu().numpy().tolist()
        sae_preds = sae_preds.float().detach().cpu().numpy().tolist()
        
        df.loc[df['label'] == t, 'sev_pred'] = sev_preds
        torch.cuda.empty_cache()
        gc.collect()


    """
    correlation analysis
    """
    # df = pd.read_csv(f'{cfg.sae_out_dir}/{cfg.df_file_name}_v{cfg.current_v}.csv')
    _df = df.copy()
    _df['severity_score'] = _df['severity'].map(sev_dict)
    _df['sev_pred'] = _df['sev_pred'].astype(float)

    n_cols, n_rows = 7,2
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 5))

    for i, t in (enumerate(act_labels)):
        
        _df_t = _df[_df['label'] == t]

        _df_t.loc[:,'sev_pred'] = _df_t['sev_pred'].map(lambda x: x / act_max_dict[t])

        row_i, col_i = divmod(i, n_cols)
        r = (statsc.spearmanr(_df_t['severity_score'], _df_t['sev_pred'])[0])
        sns.lineplot(x='severity_score', y='sev_pred', data=_df_t, ax=axes[row_i, col_i], err_style='bars', color='teal', linewidth=4, err_kws={'capsize': 5, 'capthick': 4, 'elinewidth': 4})
        axes[row_i, col_i].set_title(f"", fontweight='bold', fontsize=14)
        axes[row_i, col_i].tick_params(axis='both', which='major', labelsize=16)
        axes[row_i, col_i].set_xticks([]); #axes[row_i, col_i].set_yticks([])
        
        if row_i == n_rows - 1:
            axes[row_i, col_i].set_xlabel('', fontweight='bold', fontsize=14)
        else:
            axes[row_i, col_i].set_xlabel('')
        if col_i == 0:
            axes[row_i, col_i].set_ylabel('', fontweight='bold', fontsize=14)
        else:
            axes[row_i, col_i].set_ylabel('')
        axes[row_i, col_i].text(0.5, 0.9, f"Corr.={r:.2f}", ha='center', va='center', transform=axes[row_i, col_i].transAxes, fontsize=18, fontweight='bold')
        plt.subplots_adjust(wspace=0.25, hspace=0.35)