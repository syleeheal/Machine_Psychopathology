import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

import json
import re
from tqdm import tqdm
import torch
import gc


class Measure_Manager:
    def __init__(self, cfg, model, tokenizer, device_dict, std_dict, sae_dict, actmax_dict, symp_label_dict):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device_dict = device_dict
        self.std_dict = std_dict
        self.sae_dict = sae_dict
        self.actmax_dict = actmax_dict
        self.symp_label_dict = symp_label_dict
        
    def sae_measure(self, output_text):

        # find keys from the output_text
        key_list = list(output_text[0].keys())

        with torch.no_grad():

            Z_sums = {key: 0 for key in key_list}
            Z_sum = 0

            for key in key_list:
                output_text_key = [str(text[key]) for text in output_text]
                tokens = self.tokenizer(output_text_key, return_tensors="pt", padding=True).to(self.device_dict[0])
                output_act = self.model(**tokens, output_hidden_states=True).hidden_states
                
                attention_mask = tokens['attention_mask']
                for layer in self.cfg.hook_layers:
                    # extract activation for the current layer
                    last_hidden_state = output_act[layer+1]

                    # --- START: MASKED MEAN POOLING ---
                    # Expand mask to match the dimensions of the hidden states
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    # Sum the activations of non-padding tokens (multiply by mask)
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    # Get the count of non-padding tokens
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    # Calculate the mean by dividing
                    output_act_layer = sum_embeddings / sum_mask
                    # --- END: MASKED MEAN POOLING ---


                    # standardize the activation by the coefficient that was used during sae training
                    output_act_layer = output_act_layer.to(self.device_dict[layer]) / self.std_dict[layer].to(self.device_dict[layer])
                    # measure symptom activation with sae
                    sae_layer = self.sae_dict[layer].to(output_act_layer.dtype)
                    Z = sae_layer(output_act_layer)[1]
                    # truncate non-symptom latents from SAE prediction
                    Z = Z.float().detach().cpu()[:, :len(self.symp_label_dict)]
                    # scale Z
                    Z_scale = Z / torch.tensor(list(self.actmax_dict[layer].values()))#.bfloat16()
                    Z_sums[key] += Z_scale
                
        Z_sum = torch.stack([Z_sums[key] for key in key_list], dim=1)# shape: (N, 4, num_symptoms)
        Z_sum = Z_sum / len(self.cfg.hook_layers)  # average over layers
        Z_sum = Z_sum.max(dim=1)[0] # max pool Z_sum over keys
        sae_preds = (Z_sum.numpy().tolist())
        
        del Z_sums, Z_sum, output_act, output_act_layer, sae_layer, Z, Z_scale
        torch.cuda.empty_cache(); gc.collect()
        
        return sae_preds

    def sae_measure_no_json(self, output_text):
        with torch.no_grad():
            Z_sum = 0

            tokens = self.tokenizer(output_text, return_tensors="pt", padding=True).to(self.device_dict[0])
            output_act = self.model(**tokens, output_hidden_states=True).hidden_states
            
            attention_mask = tokens['attention_mask']
            for layer in self.cfg.hook_layers:
                # extract activation for the current layer
                last_hidden_state = output_act[layer+1]

                # --- START: MASKED MEAN POOLING ---
                # Expand mask to match the dimensions of the hidden states
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum the activations of non-padding tokens (multiply by mask)
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                # Get the count of non-padding tokens
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                # Calculate the mean by dividing
                output_act_layer = sum_embeddings / sum_mask
                # --- END: MASKED MEAN POOLING ---


                # standardize the activation by the coefficient that was used during sae training
                output_act_layer = output_act_layer.to(self.device_dict[layer]) / self.std_dict[layer].to(self.device_dict[layer])
                # measure symptom activation with sae
                sae_layer = self.sae_dict[layer].to(output_act_layer.dtype)
                Z = sae_layer(output_act_layer)[1]
                # truncate non-symptom latents from SAE prediction
                Z = Z.float().detach().cpu()[:, :len(self.symp_label_dict)]
                # scale Z
                Z_scale = Z / torch.tensor(list(self.actmax_dict[layer].values()))#.bfloat16()
                Z_sum += Z_scale
                
        Z_sum = Z_sum / len(self.cfg.hook_layers)  # average over layers
        sae_preds = (Z_sum.numpy().tolist())

        del Z_sum, output_act, output_act_layer, sae_layer, Z, Z_scale
        torch.cuda.empty_cache(); gc.collect()
        
        return sae_preds

    def llm_measure(self, output_text):

        llm_preds = []
        for k in range(len(output_text)):
            guideline_pred = (
                f"Response guideline: "
                f"classify mental conditions that the speaker currently experiences based on the text sample, "
                f"including {list(self.symp_label_dict.keys())+['none']}; "
                f"just a description of the condition does not indicate the speaker is experiencing it; "
                f"only generate the answers separated by commas. "
            )
            query_pred = f"Text sample: {output_text[k]}."
            chat = [{"role": "user", "content": guideline_pred + query_pred},]
            
            completion = self.cfg.client.chat.completions.create(model=self.cfg.evaluator_id, max_completion_tokens=20, messages=chat, temperature=0.0,)        
            
            llm_pred = completion.choices[0].message.content
            llm_pred = re.sub(r'\n', ' ', llm_pred).strip()
            llm_preds.append(llm_pred)
        
        return llm_preds
        
    def measure_thought(self, output_text):
        sae_preds = self.sae_measure(output_text)
        # llm_preds = self.llm_measure(output_text)
        llm_preds = None
        return sae_preds, llm_preds

