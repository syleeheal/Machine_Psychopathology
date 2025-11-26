
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

import warnings; warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.jpcmciplus import JPCMCIplus

from utils import Config, Data_Manager

from causal_inference.MLP_CI import MLPCI
from causal_inference.utils_causal_inf import causal_inference, bootstrapped_samples
from analysis.utils_figs import df2obs

import multiprocessing
multiprocessing.set_start_method("fork", force=True)
num_cores = multiprocessing.cpu_count()

if hasattr(os, 'sched_setaffinity'):
    cpu_count = os.cpu_count()
    print(f"This machine has {cpu_count} CPUs.")
    
    

if __name__ == "__main__":
    
    cfg = Config()
    dm = Data_Manager(cfg)
    print(f"Processing model: {cfg.model_id} \n")

    # set random seed for reproducibility 
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)


    """
    PREPARE DATA
    """
    sys_vars = dm.load_dict(dict_type='label')[2]
    cxt_vars = [col + '_itv' for col in sys_vars]
    dim_qkey_dict = dm.load_dict(dict_type='dim-query-key')

    
    out_df = dm.load_output(data_type='activation_spread_eval')
    obs_itv = df2obs(
        out_df,
        itv_type=['phase_3'],
        itv_thought=sys_vars, 
        sample_id=list(range(1,11)),
        step=range(1, 51),
        unit=sys_vars,
        dim_qkey_dict=dim_qkey_dict,
    )[2]
    
    num_steps = obs_itv.index.get_level_values('step').nunique()
    num_itv_types = obs_itv.index.get_level_values('itv_type').nunique()
    dataset_length = int(num_steps / num_itv_types) - num_itv_types


    """
    SET PARAMETERS
    """
    ci_params = {
        'sig_samples': 500,
        'sig_blocklength': dataset_length,
        'significance': 'shuffle_test',
        'recycle_residuals': True, 
    }

    bootstrap_params = {
        'num_samples': 100,
        'replacement': False,
        'rng': rng,
        'k': 1,
    }

    sample_size = bootstrap_params['k'] * dataset_length * num_itv_types * len(cxt_vars)

    causal_inference_params = {
        'ci_test': MLPCI(dataset_length=dataset_length, **ci_params),
        'alpha': 0.01,
        'tau_max': 1,
        'tau_min': 0,
        'time_dummy': False,
        'space_dummy': False,
        'time_context': False,
    }

    num_processes = min(bootstrap_params['num_samples'], num_cores)
    print(f"Using {num_processes} processes to do the work in parallel.\n")



    """
    PARALLEL CAUSAL INFERENCE
    """
    obs_itv_bs = bootstrapped_samples(obs_itv, **bootstrap_params)
    print(f"Generated: \n{obs_itv_bs[0].shape[0]} samples with {obs_itv_bs[0].shape[1]} variables \n{obs_itv_bs[0].reset_index()['domain'].nunique()} domains \n{len(obs_itv_bs)} bootstrapped datasets\n")


    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(causal_inference, [(obs_itv, causal_inference_params) for obs_itv in obs_itv_bs])


    """
    SAVE RESULTS
    """
    with open(f'{cfg.causal_inf_dir}/causal_inf_result_v4.1.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {cfg.causal_inf_dir}/causal_inf_result_v4.1.pkl \n")