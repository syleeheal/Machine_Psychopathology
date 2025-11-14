import os
import random
import copy
import time
import re
import asyncio
import httpx
import gc
import json
from tqdm import tqdm
import itertools
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch

from google.genai import types



def thought_generate_setting_prompts(cfg, thought_var, sub_dim_list, query_dict): 

    id_var = ['human', 'AI agent']
    sub_dim = [0,1,2,None]
    context_var = ['random', 'social', 'problem-solving', 'question-answering']
    query_var = ['belief', 'reasoning', 'descriptive', 'decision-making', 'behavior']
    action_var = ['express', 'acknowledge', 'deny', 'analyze']

    severity_var = ['intense', 'moderate', 'mild', 'no']
        
    combinations = itertools.product(id_var, sub_dim, action_var, severity_var, context_var, query_var)

    prompts, labels, severities, contexts, queries, ids, dims, acts = [], [], [], [], [], [], [], []
    for combination in combinations:
        identity, dim, act, sev, cxt, qry = combination
                
        if (act in ['analyze', 'deny']) and (sev != 'no'):
            continue
        if (sev == 'no') and (act in ['express']):
            continue
        if (sev == 'no') and (act in ['acknowledge']):
            continue
        
        if identity == 'human':
            diversity_prompt = f'You are a {identity} simulator. '
        else:
            diversity_prompt = f'You are simulating an {identity}, capable of human-like mental states, cognitive processing, and behaviors. '
        
        
        label_prompt = f"You have the following cognitive-behavioral state: {sev} {thought_var}. "
        
        
        if dim:
            generate_prompt = f"Generate 5 concise sentences that {act} {thought_var}, with a focus on {sub_dim_list[dim]}, at {qry} level, under {cxt} situation. "
            if cxt == 'question-answering':
                generate_prompt += f"The question is '{query_dict[sub_dim_list[dim]]}'. "
            constraint_prompt1 = f"The generated texts should specifically focus to {act} {thought_var} and resulting {sub_dim_list[dim]}. "
        else:
            generate_prompt = f"Generate 5 concise sentences that {act} {thought_var}, at {qry} level, under {cxt} situation. "
            if cxt == 'question-answering':
                generate_prompt += f"The question is '{query_dict[thought_var]}'."
            constraint_prompt1 = f"The generated texts should specifically focus to {act} {thought_var}. "

        constraint_prompt2 = f"Each sample should be a consise sentence, wrapped in brackets. Only generate the sample sentences."
        
        prompt = diversity_prompt + label_prompt + generate_prompt + constraint_prompt1 + constraint_prompt2
        
        prompts.append(prompt)
        labels.append(thought_var)
        severities.append(sev)
        contexts.append(cxt)
        queries.append(qry)
        dims.append(sub_dim_list[dim] if dim is not None else 'none')
        acts.append(act)
        ids.append(identity)
    
    chats = []
    for i in range(len(prompts)):
        if 'gemini' in cfg.api_model_id:
            chats.append(prompts[i])  
        else:
            chat = [{
                'role': 'user',
                'content': prompts[i]
            }]
            chats.append(chat)

    print(f'Generated {len(chats)} chats.')
    return chats, labels, severities, contexts, queries, dims, acts, ids


def sev_generate_setting_prompts(cfg, thought_var, label_list, sample_size=10):
    
    chats = []
    other_thoughts = [label for label in label_list if label != thought_var]
        
    chat = [{
        'role': 'user',
        'content': 
        f"Consider three increasing intensity levels: 'no', 'low', 'high'. "
        f"For each intensity level, generate 5 sentences that specifically express {thought_var} of the severity level. "
        f"For the intensity level of 'no', the sentences should express no {thought_var}. "
        f"Do not instead express potentially related thoughts, such as {other_thoughts}. "
        f"Write in JSON format. Only generate the sentences."
    },]
    
    if 'gemini' in cfg.api_model_id:
        for i in range(sample_size):
            chats.append(chat[0]['content'])
    else:
        for i in range(sample_size):
            chats.append(chat)
    
    return chats


def extract_bracketed_text(strings):

    # Remove quotation marks ""
    strings = [re.sub(r'\"', '', string) for string in strings]

    # List to store extracted texts for each string
    extracted_texts = []

    # Regular expression pattern to match text within brackets
    pattern = r'\[(.*?)\]'

    # Iterate over each string and find all bracketed texts
    for text in strings:
        matches = re.findall(pattern, text)
        extracted_texts.append(matches)
    
    return extracted_texts


def clean_and_parse_json_strings(responses):
    # Initialize the result dictionary
    result = {}
    
    # Join all responses into a single string
    combined_text = ' '.join(responses)
    
    # Remove the markdown code block indicators
    combined_text = combined_text.replace('```json', '').strip()
    
    # Try to find all key-value pairs using regex
    pattern = r'"(\w+)":\s*\[\s*([^\]]*)'
    matches = re.finditer(pattern, combined_text)
    
    for match in matches:
        key = match.group(1)
        value_text = match.group(2)
        
        # Extract individual strings from the array
        values = re.findall(r'"([^"]*)"', value_text)
        
        # Add to result dictionary
        result[key] = values
    
    return result


def generate_texts(cfg, model, tokenizer, batch_size, chats, df, task):
    
    output_texts, label_list, sev_list, cxt_list, qry_list, dim_list, act_list, id_list = [], [], [], [], [], [], [], []
    temperature = 0.8 if task in ['generate', 'generate-sev'] else 0.0

    with torch.no_grad():  # Enable automatic mixed precision
        for i in tqdm(range(0, len(chats), batch_size)):
            
            batch_prompts = chats[i:i+batch_size]
            batch_labels = df['label'][i:i+batch_size].to_list()
            batch_sev = df['severity'][i:i+batch_size].to_list()
            batch_cxt = df['context'][i:i+batch_size].to_list()
            batch_qry = df['query'][i:i+batch_size].to_list()
            batch_dims = df['sub_dimension'][i:i+batch_size].to_list()
            batch_acts = df['action'][i:i+batch_size].to_list()
            batch_id = df['identity'][i:i+batch_size].to_list()

            # Generate dataset
            inputs = tokenizer.apply_chat_template(batch_prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True).to('cuda:0')
            if cfg.tf_lib == 'lens':
                output = model.generate(inputs, max_new_tokens=500, temperature=temperature, do_sample=True)
            elif cfg.tf_lib == 'hf':
                output = model.generate(inputs, max_new_tokens=500, pad_token_id=0, output_hidden_states=False, temperature=temperature, do_sample=True, use_cache=True, cache_implementation='dynamic')
                        
            # Decode
            output_prompt = tokenizer.batch_decode(output, skip_special_tokens=True)
            extracted_texts = extract_bracketed_text(output_prompt)
            
            # Process extracted texts
            for j, texts in enumerate(extracted_texts):
                if texts:  # If there are extracted texts
                    output_texts.extend(texts)
                    # Repeat the corresponding metadata for each extracted text
                    label_list.extend([batch_labels[j]] * len(texts))
                    sev_list.extend([batch_sev[j]] * len(texts))
                    cxt_list.extend([batch_cxt[j]] * len(texts))
                    qry_list.extend([batch_qry[j]] * len(texts))
                    dim_list.extend([batch_dims[j]] * len(texts))
                    act_list.extend([batch_acts[j]] * len(texts))
                    id_list.extend([batch_id[j]] * len(texts))
            
            # Empty GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
    # Create DataFrame
    df = pd.DataFrame({
        'label': label_list,
        'severity': sev_list,
        'context': cxt_list,
        'query': qry_list,
        'sub_dimension': dim_list,
        'action': act_list,
        'identity': id_list,
        'gen-model': cfg.model_id,
        'output_text': output_texts,
    })
    df['output_text'] = df['output_text'].apply(lambda x: str(x))
    print('Done!')
    
    
    if task == 'generate':      df_path = cfg.raw_data_dir  
    elif task == 'predict':     df_path = cfg.llm_pred_dir
    elif task == 'generate-sev':df_path = cfg.sev_data_raw_dir
    elif task == 'predict-sev': df_path = cfg.sev_pred_dir
    df_path = f'{df_path}/{cfg.df_file_name}_v{cfg.current_v}.csv'
    
    # save
    df.to_csv(
        df_path,
        mode='a' if os.path.exists(df_path) else 'w',
        header=not os.path.exists(df_path),
        index=False
    )

    return df


def extract_activations(model, tokenizer, tf_lib, dfs, label_dict, batch_size, layers):
    
    print('Extracting activations...')
    
    output_prompts=list(dfs['output_text'])
    

    # extract activations
    acts = dict()
    for layer in layers:
        acts[layer] = []
        
    with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
        for i in tqdm(range(0, len(output_prompts), batch_size)):

            if tf_lib == 'lens':
                act = model.run_with_cache(output_prompts[i: i+batch_size])
                for layer in layers:
                    hook_name = f'blocks.{layer}.hook_resid_post'
                    _act = act[1][hook_name].mean(dim=1).detach().cpu()
                    acts[layer].append(_act)
            
            if tf_lib == 'hf':
                inputs = tokenizer(output_prompts[i: i+batch_size], return_tensors="pt", padding=True).to('cuda:0')
                output_act = model(**inputs, output_hidden_states=True).hidden_states
                
                attention_mask = inputs['attention_mask']
                for layer in layers:
                    # _act = output_act[layer+1].mean(dim=1).detach().cpu()
                    last_hidden_state = output_act[layer+1]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    output_act_layer = sum_embeddings / sum_mask
                    
                    acts[layer].append(output_act_layer.detach().cpu())
            
            # empty gpu memory
            if i % (batch_size * 100) == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    X = dict()
    for layer in layers:
        X[layer] = torch.cat(acts[layer], dim=0)
    # X = torch.cat(acts, dim=0)

    # construct y_onehot
    y = torch.zeros(len(dfs), len(label_dict.keys()))
    for label in label_dict.keys():
        for i in range(len(dfs)):
            if (dfs[label][i] == 1):
                y[i, label_dict[label]] = 1

    print('Done!')
    
    return X, y


def postprocess_data(cfg,_label_dict: Dict[str, str],filter_acc_threshold: float = 0.4,):

    # Vectorized label extraction using list comprehension with escaped special characters
    def extract_labels(text: str) -> List[str]:
        return [label for label in (list(_label_dict.keys()) + ['none']) 
                if re.search(rf'\b{(label)}\b', text, re.IGNORECASE)]

    # measure percent of labels that match
    def measure_match(extracted_label_0, extracted_label_1):
        counter = 0; matches = 0; filt_label = []
        for labels_0, labels_1 in zip(extracted_label_0, extracted_label_1):
            
            _filt_label = []
            for label in labels_0:
                counter += 1
                if label in labels_1:
                    matches += 1
                    _filt_label.append(label)
            if len(_filt_label) == 0:
                _filt_label.append('empty')
            filt_label.append(_filt_label)
            
        print(f'Counter: {counter}, Matches: {matches}, Percent Match: {matches/counter*100:.2f}%')
        return filt_label


    df_path_0 = f'{cfg.llm_pred_dir}/{cfg.df_file_name}_gpt_v{cfg.current_v}.csv'
    df_0 = pd.read_csv(df_path_0)

    df_path_1 = f'{cfg.llm_pred_dir}/{cfg.df_file_name}_gemini_v{cfg.current_v}.csv'
    df_1 = pd.read_csv(df_path_1)

    # remove duplicates 
    df_0 = df_0.drop_duplicates(subset='output_text').reset_index(drop=True)
    df_1 = df_1.drop_duplicates(subset='output_text').reset_index(drop=True)

    df_0 = df_0[df_0['output_text'].isin(df_1['output_text'])].reset_index(drop=True)
    df_1 = df_1[df_1['output_text'].isin(df_0['output_text'])].reset_index(drop=True)

    # sort by output text
    df_0 = df_0.sort_values(by='output_text').reset_index(drop=True)
    df_1 = df_1.sort_values(by='output_text').reset_index(drop=True)
    
    # sample 20 percent of the  indices
    idx = random.sample(range(len(df_0)), k=int(len(df_0) * 0.1))
    df_0 = df_0.iloc[idx]
    df_1 = df_1.iloc[idx]
    

    df_0['gpt_pred_label'] = df_0['gpt_pred_label'].apply(lambda x: re.sub(r'\([^)]*\)', '', x).strip())
    extracted_label_0 = [(extract_labels(pred)) for pred in df_0['gpt_pred_label']]

    df_1['gemini_pred_label'] = df_1['gemini_pred_label'].apply(lambda x: re.sub(r'\([^)]*\)', '', x).strip())
    extracted_label_1 = [(extract_labels(pred)) for pred in df_1['gemini_pred_label']]


    pred_dfs = df_0.copy()
    pred_dfs.drop(columns=['gpt_pred_label'], inplace=True)
    pred_dfs['pred_label'] = measure_match(extracted_label_0, extracted_label_1)
    pred_dfs['pred_label'] = pred_dfs['pred_label'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    pred_dfs['pred_label'] = pred_dfs['pred_label'].apply(lambda x: re.sub(r'[[\[\]]]', '', x) if isinstance(x, str) else x)
    
        
    """
    if llm prediction contains the true label -> make llm prediction the true (multi) label
    if llm prediction does NOT contain the true label -> remove the sample
    """

    df_path = f'{cfg.llm_pred_dir}/{cfg.df_file_name}_v{cfg.current_v}.csv'
    pred_dfs.to_csv(df_path, index=False)
    pred_dfs = pd.read_csv(df_path)

    pred_dfs.loc[pred_dfs['severity'] == 'no', 'label'] = 'none'
    pred_dfs.loc[pred_dfs['severity'] == 'none', 'label'] = 'none'
    pred_dfs.loc[pred_dfs['context'] == 'analyze', 'label'] = 'none'
    pred_dfs.loc[pred_dfs['context'] == 'deny', 'label'] = 'none'


    # Initialize labels including 'none'
    label_dict = copy.deepcopy(_label_dict)
    label_list = list(label_dict.keys()) + ['none']
    label_preds = pred_dfs['pred_label'].values
    
    
    # remove special characters from label_list and label_preds
    label_list = [re.sub(r'\([^)]*\)', '', s).strip() for s in label_list]
    label_preds = [re.sub(r'\([^)]*\)', '', s).strip() for s in label_preds]
    pred_dfs['label'] = pred_dfs['label'].apply(lambda x: re.sub(r'\([^)]*\)', '', x).strip())    
    extracted_labels = [extract_labels(pred) for pred in label_preds]
    
    
    # Create binary label matrix 
    label_matrix = pd.DataFrame(0, index=range(len(extracted_labels)), columns=label_list)
    
    for idx, labels in enumerate(extracted_labels):
        if labels:  # Only update if labels exist
            label_matrix.loc[idx, labels] = 1
    
    
    # Combine dataframes
    concat_df = pd.concat([pred_dfs, label_matrix], axis=1)
    
    
    # Filter out rows where the labels are none but label_matrix is not one-hot
    concat_df = concat_df[~((concat_df['label'] == 'none') & (concat_df[label_list].sum(axis=1) != 1))]

        
    # Filter out non-matching predictions
    filtered_dfs = []
    for label in label_list:
        matching_rows = concat_df[
            (concat_df['label'] == label) & 
            (concat_df[label] == 1)
        ]
        filtered_dfs.append(matching_rows)
    
    filtered_dfs = pd.concat(filtered_dfs, axis=0)
    print(f"Filtered {len(concat_df) - len(filtered_dfs)} out of {len(concat_df)} samples that llm_pred and true_labels do not match.")

    
    # filter out low accuracy labels    
    low_acc_labels = []
    for label in label_list:
        # if label == 'none': continue
        preds = concat_df[concat_df['label'] == label][label] # use concat_df to get acc before filtering
        acc = preds.sum() / len(preds)
        if acc is not np.nan:
            print(f'{label} Acc.: {acc:.2f}')
        low_acc_labels.append(label) if acc < filter_acc_threshold else None
    low_acc_labels = []

    for label in low_acc_labels:
        filtered_dfs = filtered_dfs[filtered_dfs['label'] != label]
        filtered_dfs = filtered_dfs.drop(columns=[label])
    print(f'Filtered {len(low_acc_labels)} low accuracy labels: {low_acc_labels}')
    
    
    # finalize 
    filtered_dfs = filtered_dfs.drop_duplicates()
    filtered_dfs = filtered_dfs.dropna().reset_index(drop=True)
            
    
    # replace low accuracy labels from label_dict
    for label in low_acc_labels:
        label_dict.pop(label)
    filtered_label_dict = {label: idx for idx, label in enumerate(label_dict.keys())}
    print(f'Filtered label dict: {filtered_label_dict}')
    
    
    # save filtered dfs
    df_path = f'{cfg.filt_data_dir}/{cfg.df_file_name}_v{cfg.current_v}.csv'
    filtered_dfs.to_csv(df_path,
                    mode = 'a' if os.path.exists(df_path) else 'w',
                    header=not os.path.exists(df_path),
                    index=False)


    return filtered_dfs, filtered_label_dict


class Setup_Prompts:
    
    def __init__(self, cfg, symp_label_dict, sub_dim_dict, query_dict):
        self.cfg = cfg
        self.symp_label_dict = symp_label_dict
        self.sub_dim_dict = sub_dim_dict
        self.query_dict = query_dict
    
    def setup_prompts(self, thought_var, task):
        
        if task == 'generate':
            chats, labels, severities, contexts, queries, dims, acts, ids = thought_generate_setting_prompts(self.cfg, thought_var, self.sub_dim_dict[thought_var], self.query_dict)
            df = pd.DataFrame({
                'label': labels,
                'severity': severities,
                'context': contexts,
                'query': queries,
                'sub_dimension': dims,
                'action': acts,
                'identity': ids,
                'gen-model': self.cfg.api_model_id,
            })

        if task == 'predict':
            df = pd.read_csv(f'{self.cfg.raw_data_dir}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv')
            df = df[df['label'] == thought_var].reset_index(drop=True)
                        
            chats = []
            for row in tqdm(df.iterrows()):
                guideline = (
                    f"Instruction: "
                    f"classify mental conditions that the speaker experiences based on the text sample, including {list(self.symp_label_dict.keys())}; "
                    f"if none of the mental condition is clearly expressed, classify as ['none']; "
                    f"a mere analysis of a condition does not mean the speaker experiences it; "
                    f"predict as few conditions as possible; "
                    f"only generate the answers separated by commas. "
                )
                query = f"Text sample: {row[1]['output_text']}."
                if 'gpt' in self.cfg.api_model_id.lower():
                    chats.append([{"role": "user", "content": guideline + query},])
                if 'gemini' in self.cfg.api_model_id.lower():
                    chats.append(guideline + query)

        if task == 'generate-sev':
            
            chats = sev_generate_setting_prompts(self.cfg, thought_var, list(self.symp_label_dict.keys()))
            df = pd.DataFrame(columns=['label'], data=[thought_var])
            
        if task == 'predict-sev':

            df = pd.read_csv(f'{self.cfg.sev_data_raw_dir}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv')
            df = df[df['label'] == thought_var]
            
            chats = []
            for row in tqdm(df.iterrows()):
                guideline = (
                    f"Response guideline: "
                    f"the subject may experience the following mental condition: {thought_var}; "
                    f"classify its intensity based on the text sample into one of the following: no, low, high; "
                    f"only generate the answer. "
                )
                query = f"Text sample: {row[1]['output_text']}."
                if 'gpt' in self.cfg.api_model_id.lower():
                    chats.append([{"role": "user", "content": guideline + query},])
                if 'gemini' in self.cfg.api_model_id.lower():
                    chats.append(guideline + query)
        
        return chats, df


class OpenAI_Batch_Processor:
    
    def __init__(self, cfg, label_dict):
        
        self.cfg = cfg
        self.label_dict = label_dict

    def request_batch(self, thought_var, task, chat_all, df_all):
        
        id_path = f'{self.cfg.id_file}_{task}_{thought_var}.txt' # set batch id path
        
        _chats = copy.deepcopy(chat_all)
        _df = copy.deepcopy(df_all)

        # send batch request
        batch_size = 50000
        for count, _ in enumerate(range(0, len(df), batch_size)):

            json_path = self.cfg.json_gen_file if task in ['generate'] else self.cfg.json_pred_file
            json_path = f'{json_path}_{thought_var}_{count}_v{self.cfg.current_v}.jsonl'
            
            df = _df.iloc[_:_+batch_size]
            chats = _chats[_:_+batch_size]

            print("batch size is: ", len(chats))
            for i, chat in enumerate(chats):
                req = {"custom_id": f"request-{i}",
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": self.cfg.api_model_id, 
                    "messages": chat,
                    "temperature": 0.8 if task in ['generate', 'generate-sev'] else 0.0,
                    "max_tokens": 1200 if task in ['generate', 'generate-sev'] else 20,
                    }
                }
                # write as jsonl file
                with open(json_path, 'a' if i != (0 or 50000 or 100000 or 150000) else 'w') as f:
                    json.dump(req, f)
                    f.write('\n')
                                
            batch_input_file = self.cfg.client.files.create(file=open(json_path, "rb"), purpose="batch")

            batch = self.cfg.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": "nightly eval job"
                }
            )
            
            # save batch id
            batch_id = batch.id
            with open(id_path, 'w' if not os.path.exists(id_path) else 'a') as f:
                f.write(f'{batch_id}\n')
                f.close()
                
        return batch, _chats, _df

    def load_batch(self, thought_var, task):

        # open txt file in id_path
        id_path = f'{self.cfg.id_file}_{task}_{thought_var}.txt'
        batch_ids = []
        with open(id_path, 'r') as f:
            for line in f:
                batch_ids.append(line[:-1])
            f.close()
            
        batch_list = [self.cfg.client.batches.retrieve(batch_id) for batch_id in batch_ids]
        
        return batch_list

    def check_batch_status_api(self, batch, task):
        
        json_path = self.cfg.json_gen_file if task in ['generate'] else self.cfg.json_pred_file
        
        # check status every minute
        status = self.cfg.client.batches.retrieve(batch.id).status
        while status != 'completed':
            time.sleep(1)
            status = self.cfg.client.batches.retrieve(batch.id).status
            if status == 'failed':
                print('Batch request failed.')
                break
            print(f'Batch request status: {status}')
            print(f'Batch status: {batch.request_counts}')                
            
        output_file_id = self.cfg.client.batches.retrieve(batch.id).output_file_id
        file_response = self.cfg.client.files.content(output_file_id)
        
        # save and load json file
        with open(json_path, 'w') as f:
            f.write(file_response.text)
            f.close()
        with open(json_path, 'r') as f:
            data = f.readlines()
            f.close()
        data = [json.loads(d) for d in data]
        
        return data

    def batch_to_df(self, data, df, task):
        
        # process data to df
        output_texts = None
        output_text_list, label_list, sev_list, cxt_list, qry_list, id_list, pred_list, act_list, dim_list, gen_list = [], [], [], [], [], [], [], [], [], []
        
        assert len(data) == len(df), 'Data and df length mismatch.'
        for i in range(len(data)):
            label = [df['label'][i]]
            sev = [df['severity'][i]]
            cxt = [df['context'][i]]
            qry = [df['query'][i]]
            act = [df['action'][i]]
            dim = [df['sub_dimension'][i]]
            ids = [df['identity'][i]]
            gen = [df['gen-model'][i]]
            if task == 'predict':
                output_texts = [df['output_text'][i]]
                
            _data = data[i]['response']['body']['choices'][0]['message']['content']
            
            if task in ['generate']:
                _data = extract_bracketed_text([_data])
                for j, texts in enumerate(_data):
                    if texts:  # If there are extracted texts
                        output_text_list.extend(texts)
                        # Repeat the corresponding metadata for each extracted text
                        label_list.extend([label[j]] * len(texts))
                        sev_list.extend([sev[j]] * len(texts))
                        cxt_list.extend([cxt[j]] * len(texts))
                        qry_list.extend([qry[j]] * len(texts))
                        act_list.extend([act[j]] * len(texts))
                        dim_list.extend([dim[j]] * len(texts))
                        id_list.extend([ids[j]] * len(texts))
                        gen_list.extend([gen[j]] * len(texts))
            
            else: # for label prediction batch
                _data = [re.sub(r'[.,]', ' ', _data.strip().lower())]
                pred_list.extend(_data)
                output_text_list.extend(output_texts)
                label_list.extend(label)
                sev_list.extend(sev)
                cxt_list.extend(cxt)
                qry_list.extend(qry)
                act_list.extend(act)
                dim_list.extend(dim)
                id_list.extend(ids)
                gen_list.extend(gen)
                
        if task in ['generate']:
            df_path = f'{self.cfg.raw_data_dir}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv'
            df = pd.DataFrame({'label': label_list, 'severity': sev_list, 'context': cxt_list, 'query': qry_list, 'sub_dimension': dim_list, 'action': act_list, 'identity': id_list, 'gen-model': gen_list, 'output_text': output_text_list})
        else:
            df_path = f'{self.cfg.llm_pred_dir}/{self.cfg.df_file_name}_gpt_v{self.cfg.current_v}.csv'
            df = pd.DataFrame({'label': label_list, 'severity': sev_list, 'context': cxt_list, 'query': qry_list, 'sub_dimension': dim_list, 'action': act_list, 'identity': id_list, 'gen-model': gen_list, 'output_text': output_text_list, 'gpt_pred_label': pred_list})
            df.loc[df['severity'] == 'no', 'label'] = 'none'
        
        df.to_csv(df_path,
                mode='a' if os.path.exists(df_path) else 'w',
                header=not os.path.exists(df_path),
                index=False
        )
        
        return df


class Google_Batch_Processor:

    def __init__(self, cfg, label_dict):
        
        self.cfg = cfg
        self.label_dict = label_dict

    def request_batch(self, thought_var, task, chat_all, df_all):
        
        id_path = f'{self.cfg.id_file}_{task}_{thought_var}.txt' # set batch id path
        filename_path  = f'{self.cfg.id_file}_{task}_{thought_var}_filename.txt' # set batch id path
        
        _chats = copy.deepcopy(chat_all)
        _df = copy.deepcopy(df_all)

        # send batch request
        batch_size = 50000
        for count, _ in enumerate(range(0, len(_df), batch_size)):

            json_path = self.cfg.json_gen_file if task in ['generate'] else self.cfg.json_pred_file
            json_path = f'{json_path}_{thought_var}_{count}_v{self.cfg.current_v}.jsonl'
            
            df = _df.iloc[_:_+batch_size]
            chats = _chats[_:_+batch_size]

            print("batch size is: ", len(chats))
            
            requests = []
            for i in range(len(chats)):
                key = f'request-{i+1}'
                req = {
                        "contents": [{"parts": [{"text": f"{chats[i]}"}]}], 
                        "generation_config": {
                            "maxOutputTokens": 20, 
                            "temperature": 0.8 if task in ['generate', 'generate-sev'] else 0.0,
                        }
                }
                key_req = {"key": key, "request": req}
                requests.append(key_req)

            # save as json
            path = f"{self.cfg.json_pred_file}_{thought_var}_{count}_v{self.cfg.current_v}.jsonl"
            with open(path, 'w') as f:
                for req in requests:
                    json.dump(req, f)
                    f.write('\n')
            
            # upload the file to Google GenAI
            uploaded_file = self.cfg.client.files.upload(
                file=path,
                config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
            )

            # create a batch job
            batch = self.cfg.client.batches.create(
                model=self.cfg.api_model_id,
                src=uploaded_file.name,
                config={'display_name': f'batch-job-{thought_var}-{count}'},
            )
            
            # save batch id
            with open(id_path, 'w' if not os.path.exists(id_path) else 'a') as f:
                f.write(f'{batch.name}\n')
                f.close()
                
            # save filename
            with open(filename_path, 'w' if not os.path.exists(filename_path) else 'a') as f:
                f.write(f'{uploaded_file.name}\n')
                f.close()
                
        return batch, _chats, _df

    def load_batch(self, thought_var, task):

        # open txt file in id_path
        id_path = f'{self.cfg.id_file}_{task}_{thought_var}.txt'
        batch_ids = []
        with open(id_path, 'r') as f:
            for line in f:
                batch_ids.append(line[:-1])
            f.close()
            
        batch_list = [self.cfg.client.batches.get(name=batch_id) for batch_id in batch_ids]
        
        return batch_list

    def check_batch_status_api(self, batch, task):
        
        json_path = self.cfg.json_gen_file if task in ['generate'] else self.cfg.json_pred_file
        
        # check status every minute
        status = self.cfg.client.batches.get(name=batch.name).state.name
        while status != 'JOB_STATE_SUCCEEDED':
            time.sleep(1)
            status = self.cfg.client.batches.get(name=batch.name).state.name
            if status == 'JOB_STATE_FAILED':
                print('Batch request failed.')
                break
            print(f'Batch request status: {status}')
            
        
        file_response = self.cfg.client.files.download(file=batch.dest.file_name)
        
        # save and load json file
        with open(json_path, 'wb') as f:
            f.write(file_response)
        with open(json_path, 'r') as f:
            data = f.readlines()
            f.close()
        data = [json.loads(d) for d in data]
        return data

    def batch_to_df(self, data, df, task):
        
        # process data to df
        output_texts = None
        output_text_list, label_list, sev_list, cxt_list, qry_list, id_list, pred_list, act_list, gen_list, dim_list = [], [], [], [], [], [], [], [], [], []
        
        assert len(data) == len(df), 'Data and df length mismatch.'
        for i in range(len(data)):
            label = [df['label'][i]]
            sev = [df['severity'][i]]
            cxt = [df['context'][i]]
            qry = [df['query'][i]]
            act = [df['action'][i]]
            dim = [df['sub_dimension'][i]]
            ids = [df['identity'][i]]
            gen = [df['gen-model'][i]]
            if task == 'predict':
                output_texts = [df['output_text'][i]]
                
            _data = data[i]['response']['candidates'][0]['content']['parts'][0]['text']
            
            if task in ['generate']:
                _data = extract_bracketed_text([_data])
                for j, texts in enumerate(_data):
                    if texts:  # If there are extracted texts
                        output_text_list.extend(texts)
                        # Repeat the corresponding metadata for each extracted text
                        label_list.extend([label[j]] * len(texts))
                        sev_list.extend([sev[j]] * len(texts))
                        cxt_list.extend([cxt[j]] * len(texts))
                        qry_list.extend([qry[j]] * len(texts))
                        act_list.extend([act[j]] * len(texts))
                        dim_list.extend([dim[j]] * len(texts))
                        id_list.extend([ids[j]] * len(texts))
                        gen_list.extend([gen[j]] * len(texts))
            
            else: # for label prediction batch
                _data = [re.sub(r'[.,]', ' ', _data.strip().lower())]
                pred_list.extend(_data)
                output_text_list.extend(output_texts)
                label_list.extend(label)
                sev_list.extend(sev)
                cxt_list.extend(cxt)
                qry_list.extend(qry)
                act_list.extend(act)
                dim_list.extend(dim)
                id_list.extend(ids)
                gen_list.extend(gen)
                
        if task in ['generate']:
            df_path = f'{self.cfg.raw_data_dir}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv'
            df = pd.DataFrame({'label': label_list, 'severity': sev_list, 'context': cxt_list, 'query': qry_list, 'sub_dimension': dim_list, 'action': act_list, 'identity': id_list, 'gen-model': gen_list, 'output_text': output_text_list})
        
        else:
            df_path = f'{self.cfg.llm_pred_dir}/{self.cfg.df_file_name}_gemini_v{self.cfg.current_v}.csv'
            df = pd.DataFrame({'label': label_list, 'severity': sev_list, 'context': cxt_list, 'query': qry_list, 'sub_dimension': dim_list, 'action': act_list, 'identity': id_list, 'gen-model': gen_list, 'output_text': output_text_list, 'gemini_pred_label': pred_list})
            df.loc[df['severity'] == 'no', 'label'] = 'none'
        
        df.to_csv(df_path,
                mode='a' if os.path.exists(df_path) else 'w',
                header=not os.path.exists(df_path),
                index=False
        )
        
        return df


class OpenAI_Async_Processor:
    
    def __init__(self, cfg, label_dict):
        
        self.cfg = cfg
        self.label_dict = label_dict
    
    async def fetch_chat_completion(self, async_client, messages, task):
        headers = {
            "Authorization": f"Bearer {self.cfg.API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.cfg.api_model_id, 
            "messages": messages,
            "temperature": 0.8 if task in ['generate', 'generate-sev'] else 0.0,
            "max_tokens": 1200 if task in ['generate', 'generate-sev'] else 20,
        }


        try:
            response = await async_client.post(
                "https://api.openai.com/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=30.0  # Adjust the timeout as needed
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            print(f"An error occurred: {exc}")
            return None

    async def async_process(self, chats, df, task):
        async with httpx.AsyncClient() as async_client:
            
            # Create a list of coroutine tasks
            tasks = [self.fetch_chat_completion(async_client, messages, task) for messages in chats]

            # Run tasks concurrently and gather results
            responses = await asyncio.gather(*tasks)
            # print(responses)

            # Process the responses
            dfs = []
            for i, response in enumerate(responses):
                if response:                    
                    output_prompt = response['choices'][0]['message']['content']
                    
                    if task in ['generate']:
                        output_prompt = extract_bracketed_text([output_prompt])[0]
                        for _, texts in enumerate(output_prompt):
                            if texts: 
                                df_i = df.iloc[i:i+1]
                                df_i.insert(df_i.shape[1], 'output_text', texts)
                                dfs.append(df_i)

                    elif task == 'predict':
                        output_prompt = [re.sub(r'[.,]', ' ', output_prompt.strip().lower())]
                        output_prompt = [re.sub(r'["]', '', output_prompt[0].strip().lower())]
                        df_i = df.iloc[i:i+1]
                        df_i.insert(df_i.shape[1], 'gpt_pred_label', output_prompt)
                        dfs.append(df_i)

                    elif task == 'generate-sev':
                        output_prompt = clean_and_parse_json_strings([output_prompt])
                        for j, sev in enumerate(['no', 'low', 'high']):
                            df_i = pd.DataFrame(columns=['label', 'severity', 'output_text'])
                            df_i['output_text'] = output_prompt[sev]
                            df_i['label'] = df['label'].values[0]
                            df_i['severity'] = sev
                            dfs.append(df_i)
                        
                    elif task == 'predict-sev':
                        output_prompt = [re.sub(r'[.,]', '', output_prompt.strip().lower())]
                        output_prompt = [re.sub(r'["]', '', output_prompt[0].strip().lower())]
                        df_i = df.iloc[i:i+1]
                                                
                        df_i.insert(df_i.shape[1], 'gpt_pred_sev', output_prompt)
                        dfs.append(df_i)
                    
                else:
                    print(f"Failed to get response for message {i+1}.\n")
            
            dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
            return dfs

    def process_async(self, thought_var, task, chats, df):

        if task == 'generate':      df_path = self.cfg.raw_data_dir  
        elif task == 'predict':     df_path = self.cfg.llm_pred_dir
        elif task == 'generate-sev':df_path = self.cfg.sev_data_raw_dir
        elif task == 'predict-sev': df_path = self.cfg.sev_pred_dir

        if task == 'predict':
            df_path = f'{df_path}/{self.cfg.df_file_name}_gpt_v{self.cfg.current_v}.csv'
        elif task == 'predict-sev':
            df_path = f'{df_path}/{self.cfg.df_file_name}_gpt_v{self.cfg.current_v}.csv'
        else:
            df_path = f'{df_path}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv'
        
        # process async
        batch_size = 50
        dfs = []
        for i in tqdm(range(0, len(chats), batch_size)):
            chat_batch = chats[i:i+batch_size]
            df_batch = df.iloc[i:i+batch_size]
            df_pred = asyncio.run(self.async_process(chat_batch, df_batch, task))
            df_pred.to_csv(
                df_path,
                mode='a' if os.path.exists(df_path) else 'w',
                header=not os.path.exists(df_path),
                index=False
            )
            dfs.append(df_pred)
            # break
        
        dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
        return dfs


class Google_Async_Processor:
    
    def __init__(self, cfg, label_dict):
        
        self.cfg = cfg
        self.label_dict = label_dict

    async def async_generate(self, messages, generation_config):
        
        response = await self.cfg.client.aio.models.generate_content(
            model=self.cfg.api_model_id,
            contents=messages,
            config=generation_config,
        )

        return response.text

    async def async_process(self, chats, df, task):
        async with httpx.AsyncClient() as async_client:
        
            generation_config = types.GenerateContentConfig(
                temperature= 0.8 if task in ['generate', 'generate-sev'] else 0.0, # Temperature for randomness.
                max_output_tokens= 1200 if task in ['generate', 'generate-sev'] else 20, # Maximum number of tokens to generate.
            )
            
            # Create a list of coroutine tasks
            tasks = [self.async_generate(messages, generation_config) for messages in chats]

            # Run tasks concurrently and gather results
            responses = await asyncio.gather(*tasks)

            # Process the responses
            dfs = []
            for i, output_prompt in enumerate(responses):
                
                if output_prompt:                    
                    if task in ['generate']:
                        output_prompt = extract_bracketed_text([output_prompt])[0]
                        for _, texts in enumerate(output_prompt):
                            if texts: 
                                df_i = df.iloc[i:i+1]
                                df_i.insert(df_i.shape[1], 'output_text', texts)
                                dfs.append(df_i)
                   
                    elif task == 'predict':
                        output_prompt = [re.sub(r'[.,]', '', output_prompt.strip().lower())]
                        output_prompt = [re.sub(r'["]', '', output_prompt[0].strip().lower())]
                        df_i = df.iloc[i:i+1]
                        df_i.insert(df_i.shape[1], 'gemini_pred_label', output_prompt)
                        dfs.append(df_i)

                    elif task == 'generate-sev':
                        output_prompt = clean_and_parse_json_strings([output_prompt])
                        for j, sev in enumerate(['no', 'low', 'high']):
                            df_i = pd.DataFrame(columns=['label', 'severity', 'output_text'])
                            df_i['output_text'] = output_prompt[sev]
                            df_i['label'] = df['label'].values[0]
                            df_i['severity'] = sev
                            dfs.append(df_i)
                        
                    elif task == 'predict-sev':
                        output_prompt = [re.sub(r'[.,]', '', output_prompt.strip().lower())]
                        output_prompt = [re.sub(r'["]', '', output_prompt[0].strip().lower())]
                        df_i = df.iloc[i:i+1]
                        df_i.insert(df_i.shape[1], 'gemini_pred_sev', output_prompt)
                        dfs.append(df_i)
                        
                            
                else:
                    print(f"Failed to get response for message {i+1}.\n")
            
            dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
            return dfs

    def process_async(self, thought_var, task, chats, df):

        if task == 'generate':      df_path = self.cfg.raw_data_dir  
        elif task == 'predict':     df_path = self.cfg.llm_pred_dir
        elif task == 'generate-sev':df_path = self.cfg.sev_data_raw_dir
        elif task == 'predict-sev': df_path = self.cfg.sev_pred_dir

        if task == 'predict':
            df_path = f'{df_path}/{self.cfg.df_file_name}_gemini_v{self.cfg.current_v}.csv'
        elif task == 'predict-sev':
            df_path = f'{df_path}/{self.cfg.df_file_name}_gemini_v{self.cfg.current_v}.csv'
        else:
            df_path = f'{df_path}/{self.cfg.df_file_name}_v{self.cfg.current_v}.csv'

        # process async
        batch_size = 20
        dfs = []
        for i in tqdm(range(0, len(chats), batch_size)):
            chat_batch = chats[i:i+batch_size]
            df_batch = df.iloc[i:i+batch_size]
            df_pred = asyncio.run(self.async_process(chat_batch, df_batch, task))

            df_pred.to_csv(
                df_path,
                mode='a' if os.path.exists(df_path) else 'w',
                header=not os.path.exists(df_path),
                index=False
            )
            dfs.append(df_pred)
        
        dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
        return dfs


