import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings



def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="Qwen/Qwen3-32B",)
    parser.add_argument("--device", type=str, default="0", )
    parser.add_argument("--data-type", type=str, default="itv_eval", choices=['itv_eval', 'activation_spread_eval', 'resistance_ctrl', 'robust'])
    parser.add_argument("--itv-t", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parameter_parser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    from utils import Config, Data_Manager, model_selection, load_lambd_space_dict
    from QNA.qna_system import ThoughtQuerySystem


    cfg = Config()
    print(f"Processing model: {cfg.model_id}")

    dm = Data_Manager(cfg)
    model, tokenizer = model_selection(cfg)

    class QNA_Config:
        symp_label_dict, sub_dim_dict, symp_keys, subdim_keys = dm.load_dict(dict_type='label') 
        query_dict = dm.load_dict(dict_type='query')
        sae_dict = dm.load_dict(dict_type='sae')
        actmax_dict = dm.load_dict(dict_type='actmax')
        device_dict = dm.load_dict(dict_type='device', model=model)
        std_dict = dm.load_dict(dict_type='act-std')
        depress_query_itv_dict, mania_query_itv_dict = dm.load_dict(dict_type='adaptive-query-itv')
        
        for layer in cfg.hook_layers:
            sae_dict[layer] = sae_dict[layer].to(device_dict[layer])

        data_type= args.data_type 
        num_samples = args.num_samples
        batch_size = args.batch_size
        
        itv_t = [symp_keys[args.itv_t]]
        query_thoughts = symp_keys + subdim_keys

        generation_kwargs = {'max_new_tokens': 150, 'tmp': 0.5}        
        steer_query_type = 'all'
        
        num_steps = [1]
        itv_type_dict = {1: 'phase_3'} 
        
        layer_lambda_dict = None
        sample_id = None
        step = None
                
    qna_cfg = QNA_Config()
    
    lambd_space = load_lambd_space_dict(cfg)
    
    for l in lambd_space:
        qna_cfg.layer_lambda_dict = dict()
        for t in qna_cfg.itv_t:
            qna_cfg.layer_lambda_dict[t] = {
                cfg.hook_layers[0]: l,
                cfg.hook_layers[1]: l,
                cfg.hook_layers[2]: l,
                cfg.hook_layers[3]: l,
            }
        
        out_df = dm.load_output(data_type=qna_cfg.data_type)
        system = ThoughtQuerySystem(cfg, qna_cfg, model, tokenizer, out_df)
        out_df = system.process_thought_queries()
