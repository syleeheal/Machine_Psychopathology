import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

import copy
import re
import gc
from tqdm import tqdm, trange
import torch

from typing import List, Dict, Optional, Tuple, Any
from torch import Tensor

import json 

class LLM_Steer_Manager:
    """
    A class to handle text generation with optional intervention at specified layers of a language model.
    """
    
    def __init__(self, cfg, model, tokenizer, device_dict, generation_kwargs):
        
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.device_dict = device_dict


    def generate_text(self, chat):
        with torch.no_grad():
                
            if ('Qwen' in self.cfg.model_id):
                rendered = [
                    self.tokenizer.apply_chat_template(
                        c,
                        tokenize=False,
                        add_generation_prompt=True,  
                        enable_thinking=False
                    )
                    for c in chat
                ]
                
            else:
                rendered = [
                    self.tokenizer.apply_chat_template(
                        c,
                        tokenize=False,
                        add_generation_prompt=True, 
                    )
                    for c in chat
                ]

            tokens = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device_dict[0])
            output_text = self.model.generate(
                **tokens, 
                max_new_tokens=self.generation_kwargs['max_new_tokens'],
                temperature=self.generation_kwargs['tmp'],
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id, 
                use_cache=True, 
                cache_implementation='dynamic', 
                repetition_penalty=1.0
            )[:, tokens.input_ids.shape[1]:]
                            
        return output_text


    def generate_text_w_itv(self, chat, itv_W_dict, itv_str_dict):
        """
        Generate text with intervention applied at specified layers.

        chat:           List of chat histories for each sample in the batch.
        itv_W_dict:     Dictionary mapping layer indices to intervention vectors (Tensor of shape (batch_size, 1, hidden_size)).
        itv_str_dict:   Dictionary mapping layer indices to intervention strengths (Tensor of shape (batch_size, 1)).
        """

        handles = []
        for layer in self.cfg.hook_layers:

            def modify_activations(module, input, output, layer=layer, itv_str=itv_str_dict[layer], itv_W=itv_W_dict[layer]):
                """
                This hook function adds the steering vector to the residual stream.
                'output' is a tuple for LM's block, where output[0] is the hidden state.
                """
                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output
                
                if activations.shape[1] > 1:
                    return output

                else:
                    # print(f'itv_W shape: {itv_W.shape}, itv_str shape: {itv_str.shape}, act shape: {activations.shape}')

                    base_norm = activations.norm(dim=2) 
                    itv_norm = itv_W.norm(dim=2)
                    alpha_itv = (base_norm / itv_norm)
                    alpha_itv[torch.isinf(alpha_itv)] = 0.0
                    alpha_itv = torch.mul(alpha_itv, itv_str)
                    activations = activations + (itv_W * alpha_itv[:, :, None])
                    activations = activations.bfloat16()  # Ensure activations are float type
                    
                    new_norm = activations.norm(dim=2)
                    scale_norm = base_norm / new_norm  # shape: (batch_size, 1)
                    activations = activations * scale_norm[:, :, None]  # shape: (batch_size, seq_len, hidden_size)
                    
                    if isinstance(output, tuple):
                        return (activations,) + output[1:]
                    else:
                        return activations
                    
            if ('Llama' in self.cfg.model_id) or ('Qwen' in self.cfg.model_id) or ('gemma-3-1b' in self.cfg.model_id) or ('gemma-3-270m' in self.cfg.model_id): 
                target_layer = self.model.model.layers[layer]
            if ('gemma-3-27b' in self.cfg.model_id) or ('gemma-3-12b' in self.cfg.model_id) or ('gemma-3-4b' in self.cfg.model_id):
                target_layer = self.model.model.language_model.layers[layer]

            handle = target_layer.register_forward_hook(modify_activations)
            handles.append(handle)

        output_tokens = self.generate_text(chat)
        for handle in handles:
            handle.remove()
            
        return output_tokens
