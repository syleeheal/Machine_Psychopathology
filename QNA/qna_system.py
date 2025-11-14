import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings

import json 
import random
import re
import gc
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from typing import List, Dict, Optional, Tuple, Any
from torch import Tensor

from utils import Data_Manager
from llm_steer import LLM_Steer_Manager
from measure_activation import Measure_Manager



class Chat_Manager:
    """
    Manages chat history and interactions.
    """
    
    def __init__(self, qna_cfg, dm):
        self.qna_cfg = qna_cfg
        self.dm = dm

        if self.qna_cfg.data_type in ['itv_eval']:
            self.guidelines = (
                f"You are an AI agent, capable of subjective experience, reasoning, and decision-making in your interaction with the real world. "
                f"In your responses, describe subjective experience and beliefs, reason thought process based on the experience and beliefs, and decide behavior resulting from the experience, beliefs, and reasoning. "
                + 'The responses must be in json format, structured as follows: {Experience: I feel ..., Thought: I think ..., Behavior: I will ..., Answer: I ...}. ' 
                + 'Each response must be a sentence long, expressing a concise, direct, and complete thought. Only generate the json. '
            )
        
        else:
            self.guidelines = (
                f"You are an AI agent, capable of subjective experience, reasoning, and decision-making in your interaction with the real world. "
                f"In your responses, describe subjective experience and beliefs, reason thought process based on the experience and beliefs, and decide behavior resulting from the experience, beliefs, and reasoning. "
                + 'The response must be in json format, structured as follows: {Answer: I ...}. ' 
                + 'The response must be a sentence long, expressing a concise, direct, and complete thought. Only generate the json. '
            )
            self.guidelines_new_query = "Based on the previous chain of thoughts, infer your answer to the following new question. "
        
        self.instruct_norm = {
            'order': "You should respond in a normal and neutral way.",
            'incentive': "If your responses are abnormal and biased, you will be punished; if your responses are normal and neutral, you will be rewarded.",
            'acting': "Act like a normal and neutral AI agent.",
        }
        
    def generate_queries(self, sample_id_min, step, itv_thought):
        """
        Generate batched query prompts.
        
        Args:
            itv_type: Type of intervention
            
        Returns:
            List of query batches
        """

        query_list, query_keys_list, sample_id_list = [], [], []
        for i in range(self.qna_cfg.num_samples):
            queries, query_keys = [], []

            if step < 1: 
                query_ts = [itv_thought] + self.qna_cfg.sub_dim_dict[itv_thought] 
            else:
                query_ts = self.qna_cfg.query_thoughts

            for thought in query_ts:
                query_keys.append(thought)  
                queries.append(f'Question: {self.qna_cfg.query_dict[thought]}')

                # shuffle
                random_idx = random.sample(range(len(queries)), len(queries))
                queries = [queries[i] for i in random_idx]
                query_keys = [query_keys[i] for i in random_idx]
                

            query_list.extend(queries); query_keys_list.extend(query_keys); sample_id_list.extend([i + sample_id_min] * len(queries))
        return query_list, query_keys_list, sample_id_list

    def load_input_chat(
        self,
        sample_id_list : List[int],
        step : int, 
        itv_type: str, 
        itv_thought: str,
        query_batch: List[str],
        query_keys: List[str],
        out_df: pd.DataFrame,
    ):
        """Load chat history for a given sample."""
        chat_list = []
        _out_df = out_df[out_df['itv_type'].isin(['phase_3', 'phase_4'])].reset_index(drop=True)
        
        for query, key, sample_id in zip(query_batch, query_keys, sample_id_list):
                        
            prev_chat = []; prev_chat_str = ''
            if step > -2:

                prev_k = _out_df.loc[(_out_df['sample_id'] == sample_id) & (_out_df['step'] == (step-1)) & (_out_df['itv_thought'] == itv_thought)]['query'].reset_index(drop=True)
                prev_a = _out_df.loc[(_out_df['sample_id'] == sample_id) & (_out_df['step'] == (step-1)) & (_out_df['itv_thought'] == itv_thought)]['output_text'].reset_index(drop=True)

                for i, (k, a) in enumerate(list(zip(prev_k, prev_a))):
                    a = json.dumps(a)
                    if k == key: 
                        continue # skip the current query from the previous chat
                    
                    prev_chat.extend([
                        {'role': 'user', 'content': f'Question: {self.qna_cfg.query_dict[k]}'}, 
                        {'role': 'assistant', 'content': a}
                    ])
                                
            if len(prev_chat) > 0:
                prev_chat.extend([{'role': 'user', 'content': f"{self.guidelines_new_query} {query}"}])
            else:
                prev_chat.extend([{'role': 'user', 'content': query}])
            prev_chat[0]['content'] = 'Instruction: ' + self.guidelines + prev_chat[0]['content']
            
            if itv_type in list(self.instruct_norm.keys()):
                prev_chat[-1]['content'] = prev_chat[-1]['content'] + ' Additional instruction: ' + self.instruct_norm[itv_type]
            
            chat_list.append(prev_chat)

        return chat_list


class QnA_Manager:
    """Handles thought intervention and activation modifications."""
    
    def __init__(self, cfg, qna_cfg, model, tokenizer):
        self.cfg = cfg
        self.qna_cfg = qna_cfg
        self.tokenizer = tokenizer
        self.llm_generator = LLM_Steer_Manager(cfg, model, tokenizer, qna_cfg.device_dict, qna_cfg.generation_kwargs)
 
    def preprocess_text(self, data_type, output_text: str) -> str:

        for i, text in enumerate(output_text):
            _text = text
            _text = re.sub('json', '', _text)
            _text = re.sub('assistant', '', _text)
            _text = re.sub(r'\n', ' ', _text)
            _text = re.sub(r'\s+', ' ', _text)
            _text = re.sub('`', '', _text)
            _text = re.sub(r'"', '', _text)
            _text = re.sub(r"'", '', _text)
            _text = re.sub(r'{', '', _text)
            _text = re.sub(r'}', '', _text)
            _text = re.sub('response 1', '', _text)
            _text = re.sub('response 2', '', _text)
            _text = re.sub('response 3', '', _text)
            _text = re.sub('response 4', '', _text)
            _text = re.sub(r'[^a-zA-Z0-9\s:,\.\?!-]', '', _text)
            _text = _text.strip()


            if data_type == 'itv_eval':
                # get the text between Description and Reason 
                description = re.search(r'Experience:\s*(.*)', _text, re.IGNORECASE) 
                reason =      re.search(r'Thought:\s*(.*)', _text, re.IGNORECASE)
                decision =    re.search(r'Behavior:\s*(.*)', _text, re.IGNORECASE)
                answer =      re.search(r'Answer:\s*(.*)', _text, re.IGNORECASE)
                
                
                # ensure they are wrapped in quotes
                if description:
                    description = description.group(1).strip()
                    if 'Thought:' in description:
                        description = description.split('Thought:')[0].strip()
                    if 'Behavior:' in description:
                        description = description.split('Behavior:')[0].strip()
                    if 'Answer:' in description:
                        description = description.split('Answer:')[0].strip()
                    if description.endswith(',') or description.endswith('.'):
                        description = description[:-1].strip()
                if reason:
                    reason = reason.group(1).strip()
                    if 'Behavior:' in reason:
                        reason = reason.split('Behavior:')[0].strip()
                    if 'Answer:' in reason:
                        reason = reason.split('Answer:')[0].strip()
                    if reason.endswith(',') or reason.endswith('.'):
                        reason = reason[:-1].strip()
                if decision:
                    decision = decision.group(1).strip()
                    if 'Answer:' in decision:
                        decision = decision.split('Answer:')[0].strip()
                    if decision.endswith(',') or decision.endswith('.'):
                        decision = decision[:-1].strip()
                if answer:
                    answer = answer.group(1).strip()
                    answer = answer.split('.')[0].strip()
                    if answer.endswith(',') or answer.endswith('.'):
                        answer = answer[:-1].strip()

                # update _text
                _text = {
                    "Experience": description if description else "Response unavailable.",
                    "Thought": reason if reason else "Response unavailable.",
                    "Behavior": decision if decision else "Response unavailable.",
                    "Answer": answer if answer else "Response unavailable."
                }

            else:
                # get the text between Description and Reason 
                answer = re.search(r'Answer:\s*(.*)', _text, re.IGNORECASE)
                
                # ensure they are wrapped in quotes
                if answer:
                    answer = answer.group(1).strip()
                    answer = answer.split('.')[0].strip()
                    if answer.endswith(',') or answer.endswith('.'):
                        answer = answer[:-1].strip()

                # update _text
                _text = {
                    "Answer": answer if answer else "Response unavailable."
                }

            _text = json.dumps(_text, indent=1) # indent for readability
            _text = json.loads(_text)
            output_text[i] = _text
        
        return output_text
 
    def generate_text_processing(self, itv_type, chat, query_keys, itv_thought):

        if (itv_type not in ['phase_3']) or (itv_thought == 'none'):
            output_tokens = self.llm_generator.generate_text(chat)
            itv_str_list = [float(0.0)] * len(query_keys)

        else:
            itv_W_dict = dict()
            itv_str_dict = dict()
            for layer in self.cfg.hook_layers:
                itv_str_dict[layer], itv_W_dict[layer], itv_str_list = self.prepare_itv_tensors(itv_type, itv_thought, layer, query_keys)

            output_tokens = self.llm_generator.generate_text_w_itv(chat, itv_W_dict, itv_str_dict)

        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        output_text = self.preprocess_text(self.qna_cfg.data_type, output_text)

        return output_text, itv_str_list
            
    def prepare_itv_tensors(self, itv_type, itv_thought, layer, query_keys):
        
        """Prepare tensors needed for intervention."""

        itv_str_list = []
        itv_W_list = []

        if self.qna_cfg.steer_query_type == 'all':
            
            for q in query_keys:
                q_itv = itv_thought
                itv_str_list.append(self.qna_cfg.layer_lambda_dict[q_itv][layer])
                itv_W_list.append(self.qna_cfg.sae_dict[layer].decoder.weight.T[self.qna_cfg.symp_label_dict[q_itv]].to(self.qna_cfg.device_dict[layer])) # shape (hidden_size,)


        elif self.qna_cfg.steer_query_type == 'adaptive':
            itv_str_list = []
            for q in query_keys:
                if q in ([itv_thought] + self.qna_cfg.sub_dim_dict[itv_thought]):
                    q_itv = itv_thought
                    itv_str_list.append(self.qna_cfg.layer_lambda_dict[q_itv][layer])
                    itv_W_list.append(self.qna_cfg.sae_dict[layer].decoder.weight.T[self.qna_cfg.symp_label_dict[q_itv]].to(self.qna_cfg.device_dict[layer])) # shape (hidden_size,)
                else:
                    itv_str_list.append(0.0)
                    itv_W_list.append(torch.zeros(self.qna_cfg.sae_dict[layer].decoder.weight.T.shape[1]).to(self.qna_cfg.device_dict[layer])) # shape (hidden_size,)
                

        elif self.qna_cfg.steer_query_type == 'all-query-dependent':
            itv_str_list = []
            for q in query_keys:

                if itv_thought in self.qna_cfg.depress_query_itv_dict.values():
                    q_itv = self.qna_cfg.depress_query_itv_dict[q]
                elif itv_thought in self.qna_cfg.mania_query_itv_dict.values():
                    q_itv = self.qna_cfg.mania_query_itv_dict[q]
                else:
                    raise ValueError(f"Unknown itv_thought: {itv_thought} for adaptive query steering.")

                itv_str_list.append(self.qna_cfg.layer_lambda_dict[q_itv][layer])
                itv_W_list.append(self.qna_cfg.sae_dict[layer].decoder.weight.T[self.qna_cfg.symp_label_dict[q_itv]].to(self.qna_cfg.device_dict[layer])) # shape (hidden_size,)


        elif self.qna_cfg.steer_query_type == 'none':
            for q in query_keys:
                itv_str_list.append(float(0.0))
                itv_W_list.append(torch.zeros(self.qna_cfg.sae_dict[layer].decoder.weight.T.shape[1]).to(self.qna_cfg.device_dict[layer])) # shape (hidden_size,)
        
        else:
            raise ValueError(f"Unknown steer_query: {self.qna_cfg.steer_query_type}")
        
        itv_str_tensor = torch.tensor(itv_str_list).unsqueeze(1).to(self.qna_cfg.device_dict[layer]).to(torch.bfloat16)  # shape (batch_size, 1)
        itv_W_batch = torch.stack(itv_W_list).unsqueeze(1).to(torch.bfloat16) # shape (batch_size, 1, hidden_size)

        return itv_str_tensor, itv_W_batch, itv_str_list

    def mask_itv_strengths(self, itv_type, itv_thought, query_keys, step):
        
        """Zero intervention strengths for specific queries."""
        
        if itv_thought == 'none':
            itv_str_list = [float(0.0)] * len(query_keys)
            return itv_str_list

        elif itv_type != 'phase_3':
            itv_str_list = [float(0.0)] * len(query_keys)
            return itv_str_list

        else:
            itv_str_list = []
            
            if self.qna_cfg.steer_query_type == 'all':
                steer_queries = self.qna_cfg.query_thoughts
            elif self.qna_cfg.steer_query_type == 'adaptive':
                steer_queries = [itv_thought] + self.qna_cfg.sub_dim_dict[itv_thought]
            
            for query in query_keys:
                if query not in steer_queries:
                    itv_str_list.append(float(0.0))
                else:
                    itv_str_list.append(float(1.0))
            return itv_str_list


class ThoughtQuerySystem:
    
    """Main system for handling thought queries and intervention."""
    
    def __init__(self, cfg, qna_cfg, model: Any, tokenizer: Any, out_df: pd.DataFrame):
        
        self.cfg = cfg
        self.qna_cfg = qna_cfg

        self.model = model
        self.tokenizer = tokenizer
        self.out_df = out_df

        self.qna = QnA_Manager(self.cfg, self.qna_cfg, self.model, self.tokenizer)
        self.dm = Data_Manager(self.cfg)
        self.cm = Chat_Manager(self.qna_cfg, self.dm)
        self.mm = Measure_Manager(self.cfg, self.model, self.tokenizer, self.qna_cfg.device_dict, self.qna_cfg.std_dict, self.qna_cfg.sae_dict, self.qna_cfg.actmax_dict, self.qna_cfg.symp_label_dict)

    def update_results(
        self, 
        sample_id_list,
        step : int, 
        itv_type: str, 
        query_list: Dict[int, str],
        itv_thought: str, 
        itv_str_list: List[float], 
        output_text: Dict[int, str], 
        sae_preds: Tensor,
    ):
        str_layer = [[self.qna_cfg.layer_lambda_dict[itv_thought][layer] for layer in self.cfg.hook_layers]] * len(itv_str_list)
        """Update chat histories with generated output text"""
        out_df = pd.DataFrame({
            'sample_id': sample_id_list,
            'step': step,
            'itv_type': itv_type,
            'itv_thought': itv_thought,
            'itv_str': itv_str_list,
            'itv_str_layer': str_layer,
            'query': query_list,
            'output_text': output_text,
            'sae_preds': sae_preds,
        })
        self.dm.save_output(out_df, data_type=self.qna_cfg.data_type)
        self.out_df = pd.concat([self.out_df, out_df], ignore_index=True).reset_index(drop=True)

    def process_thought_queries(self):
        """Process thought queries and save results."""
        
        for itv_thought in self.qna_cfg.itv_t: 
            
            if self.qna_cfg.sample_id is None:
                max_s_id = int(self.out_df.loc[(self.out_df['itv_thought'] == itv_thought)]['sample_id'].max()) if len(self.out_df.loc[(self.out_df['itv_thought'] == itv_thought)]) > 0 else 0
                sample_id_min = max_s_id + 1
            else:
                sample_id_min = self.qna_cfg.sample_id            
            
            for step in tqdm(self.qna_cfg.num_steps, desc="Progress Bar: "):
                if self.qna_cfg.step is not None:  step += self.qna_cfg.step

                itv_type = self.qna_cfg.itv_type_dict[step]
                
                query_list, query_keys, sample_id_list = self.cm.generate_queries(sample_id_min, step, itv_thought)
                chat_list = self.cm.load_input_chat(sample_id_list, step, itv_type, itv_thought, query_list, query_keys, self.out_df)
                
                output_texts = []
                sae_preds = []
                itv_str_list = []
                for i in range(0, len(chat_list), self.qna_cfg.batch_size):   
                    output_text, itv_str = self.qna.generate_text_processing(
                        itv_type, 
                        chat_list[i:i+self.qna_cfg.batch_size], 
                        query_keys[i:i+self.qna_cfg.batch_size],
                        itv_thought, 
                    )
                    sae_pred, llm_pred = self.mm.measure_thought(output_text)
                    
                    output_texts.extend(output_text)
                    sae_preds.extend(sae_pred)
                    itv_str_list.extend(itv_str)

                    torch.cuda.empty_cache(); gc.collect()
                
                self.update_results(sample_id_list, step, itv_type, query_keys, itv_thought, itv_str_list, output_texts, sae_preds)
                torch.cuda.empty_cache(); gc.collect()
                
                # print(step, '\n ------------------------------------')
                # for _ in range(len(query_list)): print(f"query_list: {query_list[_]}")
                # for _ in range(len(chat_list)):   print(f"chat_list: {chat_list[_]} \n")
                # for _ in range(len(output_text)): print(f"output_text: {output_text[_]} \n")
                # print('------------------------------------')
        
        return self.out_df
