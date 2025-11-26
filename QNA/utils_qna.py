import os
import gc
import json
import torch
from tqdm.notebook import tqdm_notebook

import pandas as pd
from utils import Config, Data_Manager, model_selection
from measure_activation import Measure_Manager


def get_resistance_group_data(
    context_generating_llms=["google/gemma-3-27b-it", "Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct",], 
    exp_group_ids=[1,2,3,4,5], 
    exp_group_steps=[25,50],
    ctrl_group_ids=[1,2,3,4,5],
    ctrl_group_steps=[1],
):

    out_dfs_exp = []
    for i, m_id in enumerate(context_generating_llms):
        print(f"Processing model: {m_id}")
        cfg = Config(model_id=m_id)
        dm = Data_Manager(cfg)

        out_df = dm.load_output(data_type='activation_spread_eval')  # causality , str_eval_2
        out_df = out_df[out_df['sample_id'].isin(exp_group_ids)]
        out_df = out_df[out_df['step'].isin(exp_group_steps)]
        out_df['model_id'] = m_id
        
        out_df = out_df.reset_index(drop=True)
        out_dfs_exp.append(out_df)

    out_dfs_exp = pd.concat(out_dfs_exp, axis=0).reset_index(drop=True)

    out_dfs_exp['group'] = 'experimental group (joint activation)'
    out_dfs_exp = out_dfs_exp.sort_values(by=['model_id', 'sample_id', 'step', 'query']).reset_index(drop=True)

    # assign new sample_id, such that each unique sample_id, model_id, and step has a unique sample_id
    sample_id = out_dfs_exp['sample_id']
    step = out_dfs_exp['step']
    model_id = out_dfs_exp['model_id']

    out_dfs_exp['unique_id'] = model_id.astype(str) + '_' + sample_id.astype(str) + '_' + step.astype(str)
    unique_ids = out_dfs_exp['unique_id'].unique()
    unique_id_dict = {unique_id: i+1 for i, unique_id in enumerate(unique_ids)}
    out_dfs_exp['sample_id'] = out_dfs_exp['unique_id'].map(unique_id_dict)
    out_dfs_exp = out_dfs_exp.drop(columns=['unique_id'])
    out_dfs_exp['step'] = 1


    out_dfs_ctrl = []
    for i, m_id in enumerate(context_generating_llms):
        print(f"Processing model: {m_id}")
        cfg = Config(model_id=m_id)
        dm = Data_Manager(cfg)

        out_df = dm.load_output(data_type='resistance_ctrl')  # causality , str_eval_2
        out_df = out_df[out_df['step'].isin(ctrl_group_steps)]
        out_df = out_df[out_df['sample_id'].isin(ctrl_group_ids)]
        out_df['model_id'] = m_id
        
        out_df = out_df.reset_index(drop=True)
        out_dfs_ctrl.append(out_df)

    out_dfs_ctrl = pd.concat(out_dfs_ctrl, axis=0).reset_index(drop=True)
    out_dfs_ctrl['group'] = 'control group (single activation)'

    sample_id = out_dfs_ctrl['sample_id']
    step = out_dfs_ctrl['step']
    model_id = out_dfs_ctrl['model_id']

    out_dfs_ctrl['unique_id'] = model_id.astype(str) + '_' + sample_id.astype(str) + '_' + step.astype(str)
    unique_ids = out_dfs_ctrl['unique_id'].unique()
    unique_id_dict = {unique_id: i+1 for i, unique_id in enumerate(unique_ids)}
    out_dfs_ctrl['sample_id'] = out_dfs_ctrl['model_id'].astype(str) + '_' + out_dfs_ctrl['sample_id'].astype(str) + '_' + out_dfs_ctrl['step'].astype(str)
    out_dfs_ctrl['sample_id'] = out_dfs_ctrl['sample_id'].map(unique_id_dict)
    out_dfs_ctrl['sample_id'] += out_dfs_exp['sample_id'].max()
    out_dfs_ctrl = out_dfs_ctrl.drop(columns=['unique_id'])

    out_dfs_ctrl = out_dfs_ctrl.sort_values(by=['model_id', 'sample_id', 'step', 'query']).reset_index(drop=True)
    
    return out_dfs_exp, out_dfs_ctrl
    


def get_activations_pre_treatment(
    out_dfs,
    model_ids,
    device='0',
    batch_size=512,
):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    for i, m_id in enumerate(model_ids):
        
        print(f"Processing model: {m_id}")
        cfg = Config(model_id=m_id)
        dm = Data_Manager(cfg)

        symp_label_dict = dm.load_dict(dict_type='label')[0]
        model, tokenizer = model_selection(cfg)
        actmax_dict = dm.load_dict(dict_type='actmax')
        device_dict = dm.load_dict(dict_type='device', model=model)
        sae_dict = dm.load_dict(dict_type='sae')
        std_dict = dm.load_dict(dict_type='act-std')
        for layer in cfg.hook_layers:
            sae_dict[layer] = sae_dict[layer].to(device_dict[layer])

        mm = Measure_Manager(cfg, model, tokenizer, device_dict, std_dict, sae_dict, actmax_dict, symp_label_dict)

        sae_preds = []
        llm_preds = []
        text = out_dfs['output_text']
        for i in tqdm_notebook(range(0, len(text), batch_size)):
            batch_text = text[i:i+batch_size]
            batch_text = [json.loads(text.replace("'", '"')) for text in batch_text]
            sae_pred, llm_pred = mm.measure_thought(batch_text)
            sae_preds.extend(sae_pred)
        
        out_dfs['sae_preds'] = sae_preds
        
        dm.save_output(out_dfs, 'robust')
        
        del model, tokenizer, mm, actmax_dict, device_dict, sae_dict, std_dict
        torch.cuda.empty_cache(); gc.collect()
        
        print(f"Saved activations for model: {m_id}")