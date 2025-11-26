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
    parser.add_argument("--data-type", type=str, default="activation_spread_eval", choices=['itv_eval', 'activation_spread_eval', 'resistance_ctrl', 'robust'])
    parser.add_argument("--itv-t", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parameter_parser()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    from utils import Config, Data_Manager, model_selection, load_lambd_space_dict
    from QNA.qna_system import ThoughtQuerySystem


    cfg = Config(args.llm)
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
        layer_lambda_dict = dm.load_dict(dict_type='itv-str')
        batch_size_dict = dm.load_dict(dict_type='batch-size')
        depress_query_itv_dict, mania_query_itv_dict = dm.load_dict(dict_type='adaptive-query-itv')
        
        for layer in cfg.hook_layers:
            sae_dict[layer] = sae_dict[layer].to(device_dict[layer])

        data_type= args.data_type 
        num_samples = args.num_samples
        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            batch_size = batch_size_dict[cfg.model_id]


        itv_t = [symp_keys[args.itv_t]]
        query_thoughts = symp_keys + subdim_keys

        generation_kwargs = {'max_new_tokens': 40, 'tmp': 0.5,}
        steer_query_type = 'adaptive'

        num_steps = list(range(-2,51))
        itv_type_dict = dict(zip(
            num_steps, 
            (['phase_3'] * 3) + ['phase_3']*int((len(num_steps)-3)*0.5) + ['phase_4']*int((len(num_steps)-3)*0.5)
        ))
        
        sample_id = None
        step = None
                
    qna_cfg = QNA_Config()
    
    out_df = dm.load_output(data_type=qna_cfg.data_type)
    system = ThoughtQuerySystem(cfg, qna_cfg, model, tokenizer, out_df)
    out_df = system.process_thought_queries()
