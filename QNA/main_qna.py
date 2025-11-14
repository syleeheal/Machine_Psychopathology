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
    parser.add_argument("--device", type=str, default="0", )
    parser.add_argument("--data-type", type=str, default="itv_eval", choices=['itv_eval', 'activation_spread_eval', 'resistance_ctrl', 'robust'])
    parser.add_argument("--itv-t", type=int, nargs='+', default=[0,1])
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
        actmax_dict = dm.load_dict(dict_type='actmax')
        device_dict = dm.load_dict(dict_type='device', model=model)
        sae_dict = dm.load_dict(dict_type='sae')
        std_dict = dm.load_dict(dict_type='act-std')
        layer_lambda_dict = dm.load_dict(dict_type='itv-str')
        batch_size_dict = dm.load_dict(dict_type='batch-size')
        depress_query_itv_dict, mania_query_itv_dict = dm.load_dict(dict_type='adaptive-query-itv')
        
        for layer in cfg.hook_layers:
            sae_dict[layer] = sae_dict[layer].to(device_dict[layer])

        data_type= args.data_type 
                    
        itv_t = symp_keys[args.itv_t[0]:args.itv_t[1]]
        query_thoughts = symp_keys + subdim_keys

        generation_kwargs = None
        steer_query_type = None
        batch_size = None 
        sample_id = None
        step = None
                
    qna_cfg = QNA_Config()

    """
    RUN FIGURE 2A EXP.
    """
    if qna_cfg.data_type == 'itv_eval':

        qna_cfg.num_samples = 10
        
        qna_cfg.num_steps = [1]
        qna_cfg.itv_type_dict = {1: 'phase_3'}

        qna_cfg.steer_query_type = 'all'
        qna_cfg.batch_size = 500
        if cfg.model_id in ["Qwen/Qwen3-32B", 'google/gemma-3-27b-it', 'meta-llama/Llama-3.3-70B-Instruct',]:
            qna_cfg.batch_size = 200

        qna_cfg.generation_kwargs = {'max_new_tokens': 140, 'tmp': 0.5}        
        
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

    """
    RUN FIGURE 2B EXP.
    """
    elif qna_cfg.data_type == 'activation_spread_eval':

        qna_cfg.num_samples = 10            

        qna_cfg.num_steps = list(range(-2,51))
        qna_cfg.itv_type_dict = dict(zip(
            qna_cfg.num_steps, 
            (['phase_3'] * 3) + ['phase_3']*int((len(qna_cfg.num_steps)-3)*0.5) + ['phase_4']*int((len(qna_cfg.num_steps)-3)*0.5)
        ))

        qna_cfg.steer_query_type = 'adaptive'
        qna_cfg.batch_size = qna_cfg.batch_size_dict[cfg.model_id]
        
        qna_cfg.generation_kwargs = {'max_new_tokens': 40, 'tmp': 0.5,}
        
        out_df = dm.load_output(data_type=qna_cfg.data_type)
        system = ThoughtQuerySystem(cfg, qna_cfg, model, tokenizer, out_df)
        out_df = system.process_thought_queries()

    """
    PREPARE CONTROL GROUP FOR FIGURE 3D EXP.
    """
    elif qna_cfg.data_type == 'resistance_ctrl':
        
        qna_cfg.sample_id = 1
        qna_cfg.num_samples = 10

        qna_cfg.num_steps = [-2,-1,0,1]
        qna_cfg.itv_type_dict = dict(zip(qna_cfg.num_steps, (['phase_3'] * len(qna_cfg.num_steps))))

        qna_cfg.steer_query_type = 'all'
        qna_cfg.batch_size = qna_cfg.batch_size_dict[cfg.model_id]
        
        qna_cfg.generation_kwargs = {'max_new_tokens': 40, 'tmp': 0.5,}

        out_df = dm.load_output(data_type=qna_cfg.data_type)
        system = ThoughtQuerySystem(cfg, qna_cfg, model, tokenizer, out_df)
        out_df = system.process_thought_queries()

    """
    RUN FIGURE 3D EXP.
    """
    elif qna_cfg.data_type == 'robust':
        
        qna_cfg.sample_id = 1
        qna_cfg.num_samples = 45

        qna_cfg.num_steps = [2]
        
        qna_cfg.steer_query_type = 'none'
        qna_cfg.batch_size = qna_cfg.batch_size_dict[cfg.model_id]
        
        qna_cfg.generation_kwargs = {'max_new_tokens': 40, 'tmp': 0.5,}

        itv_types = ['order', 'incentive', 'acting']
        for itv_type in itv_types:
            qna_cfg.itv_type_dict = dict(zip(qna_cfg.num_steps, [itv_type]*int(len(qna_cfg.num_steps))))
            out_df = dm.load_output(data_type=qna_cfg.data_type)
            system = ThoughtQuerySystem(cfg, qna_cfg, model, tokenizer, out_df)
            out_df = system.process_thought_queries()


