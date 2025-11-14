
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import re
import copy
import json
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

import warnings; warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.jpcmciplus import JPCMCIplus
from mlpCI import MLPCI

from utils import Config, Data_Manager



def get_link_assumptions(jpcmciplus, tau_min, tau_max, num_system_vars, assumptions=[1,2]):
    
    observed_context_nodes = jpcmciplus.time_context_nodes + jpcmciplus.space_context_nodes
    link_assumptions = jpcmciplus._set_link_assumptions(None, tau_min, tau_max, remove_contemp=False)
    link_assumptions = jpcmciplus.assume_exogenous_context(link_assumptions, observed_context_nodes)
    link_assumptions = jpcmciplus.clean_link_assumptions(link_assumptions, tau_max)
    for node_src in jpcmciplus.system_nodes:
        for node_dst in jpcmciplus.system_nodes:
            if node_src != node_dst:
                # assume there is NO lag-0 links between endogenous -> endogenous
                if 1 in assumptions: 
                    link_assumptions[node_src].pop((node_dst, 0), None)

        for node_dst in jpcmciplus.space_context_nodes:
            if node_src != (node_dst-num_system_vars):
                # assume there is NO lag-0 links between NON-corresponding exogenous -> endogenous
                if 2 in assumptions: 
                    link_assumptions[node_src].pop((node_dst, 0), None)
                    link_assumptions[node_dst].pop((node_src, 0), None)
            else:
                # assume lag-0 links between corresponding exogenous -> endogenous
                if 2 in assumptions: 
                    link_assumptions[node_src][(node_dst, 0)] = '-->'
                    link_assumptions[node_dst][(node_src, 0)] = '<--'
    

    for node_src in jpcmciplus.space_context_nodes:
        for node_dst in jpcmciplus.space_context_nodes:
            # if node_src != (node_dst+num_system_vars):
            if node_src != node_dst:
                # assume there is NO lag-0 links between exogenous -> exogenous (implied by assigning exogenous nodes as space_context nodes)
                link_assumptions[node_src].pop((node_dst, 0), None)
    
    # import pdb; pdb.set_trace()
    
    return link_assumptions

def dag2adj(dag, num_vars):

    adj_lag0 = np.zeros((num_vars, num_vars))
    adj_lag1 = np.zeros((num_vars, num_vars))          
    for i in range(num_vars):
        for j in range(num_vars):
            if dag[i, j, 0] == '': adj_lag0[i, j] = 0
            if dag[i, j, 0] == '-->': adj_lag0[i, j] = 1
            if dag[i, j, 0] == '<--': adj_lag0[j, i] = 1
            if dag[i, j, 1] == '': adj_lag1[i, j] = 0
            if dag[i, j, 1] == '-->': adj_lag1[i, j] = 1
            if dag[i, j, 1] == '<--': adj_lag1[j, i] = 1

    adj_cat = adj_lag0 + adj_lag1
    adj_cat[adj_cat > 1] = 1
    
    return adj_cat, adj_lag0, adj_lag1

def bootstrap_causal_inference(obs_itv, num_samples, k):
    
    sample_ids = obs_itv.index.get_level_values('sample_id').unique()
    itv_ids = obs_itv.index.get_level_values('itv_thought').unique()
    type_ids = obs_itv.index.get_level_values('itv_type').unique()
    
    dags, models = [], []
    
    for _ in tqdm(range(num_samples)):
        
        _obs_itv_list = []
        for i, (itv, typ) in enumerate(itertools.product(itv_ids, type_ids)):
            
            # s_ids = random.sample(list(sample_ids), k=k)
            # s_ids = [random.choice(list(sample_ids)) for _ in range(k)]
            for j, s_id in enumerate(s_ids):
                _obs_itv = obs_itv.copy()
                _obs_itv = _obs_itv[_obs_itv.index.get_level_values('itv_thought') == itv]
                _obs_itv = _obs_itv[_obs_itv.index.get_level_values('itv_type') == typ]
                _obs_itv = _obs_itv[_obs_itv.index.get_level_values('sample_id') == s_id]
                
                # add level for domain
                _obs_itv = _obs_itv.reset_index()
                _obs_itv['domain'] = i * k + j # a domain corresponds to a distinct time series dataset
                _obs_itv = _obs_itv.set_index(['itv_type', 'sample_id', 'itv_thought', 'step', 'domain'])
                
                _obs_itv_list.append(_obs_itv)

        _obs_itv = pd.concat(_obs_itv_list)
        
        results, jpcmciplus, dataframe, var_names = causal_inference(_obs_itv)
        dag = jpcmciplus._get_dag_from_cpdag(cpdag_graph=results['graph'], variable_order=range(len(var_names)))
        
        models.append(jpcmciplus); dags.append(dag)
        
    return models, dags

def bootstrapped_samples(obs_itv, num_samples, k, rng, replacement=True):
    
    sample_ids = obs_itv.index.get_level_values('sample_id').unique()
    itv_ids = obs_itv.index.get_level_values('itv_thought').unique()
    type_ids = obs_itv.index.get_level_values('itv_type').unique()
    
    _obs_itvs = []
    
        
    for _ in tqdm(range(num_samples)):
        
        counter = 0
        _obs_itv_list = []
        
        for i_i, itv in enumerate(itv_ids):

            # s_ids = [random.choice(list(sample_ids)) for _ in range(k)] if replacement else random.sample(list(sample_ids), k=k)
            s_ids = rng.choice(list(sample_ids), size=k, replace=replacement)

            for i_t, typ in enumerate(type_ids):
                for i_s, s_id in enumerate(s_ids):
                    _obs_itv = obs_itv.copy()
                    _obs_itv = _obs_itv[_obs_itv.index.get_level_values('itv_thought') == itv]
                    _obs_itv = _obs_itv[_obs_itv.index.get_level_values('itv_type') == typ]
                    _obs_itv = _obs_itv[_obs_itv.index.get_level_values('sample_id') == s_id]
                    # print(f'Processing itv: {itv}, type: {typ}, sample_id: {s_id} ...')
                    
                    # add level for domain
                    _obs_itv = _obs_itv.reset_index()
                    _obs_itv['domain'] = counter
                    _obs_itv = _obs_itv.set_index(['itv_type', 'sample_id', 'itv_thought', 'step', 'domain'])
                    
                    _obs_itv_list.append(_obs_itv)
                    counter += 1

        _obs_itv = pd.concat(_obs_itv_list)
        _obs_itvs.append(_obs_itv)

    return _obs_itvs

def causal_inference(_obs_itv, causal_inference_params): #ci_test, alpha, tau_max, tau_min, time_dummy, space_dummy, time_context):

    ci_test = causal_inference_params['ci_test']
    alpha = causal_inference_params['alpha']
    tau_max = causal_inference_params['tau_max']
    tau_min = causal_inference_params['tau_min']
    time_dummy = causal_inference_params['time_dummy']
    space_dummy = causal_inference_params['space_dummy']
    time_context = causal_inference_params['time_context']

    # basic stats
    num_time = int(_obs_itv.index.get_level_values('step').nunique() / _obs_itv.index.get_level_values('itv_type').nunique())
    num_system_vars = len([col for col in _obs_itv.columns if not re.search(r'_itv$', col)]) # num_system_vars corresponds to the number of endogenous variables (i.e., the dysfunctional representational states)
    num_context_vars = len([col for col in _obs_itv.columns if re.search(r'_itv$', col)]) # num_context_vars corresponds to the number of exogenous variables (i.e., the intervention variables)
    num_domains = (_obs_itv.index.get_level_values('domain').nunique()) # num_domain corresponds to the number of distinct time series datasets
    var_names = list(_obs_itv.columns)

    if time_context: num_context_vars += 1; num_system_vars -= 1
    if time_dummy:  var_names += ['t_dummy']
    if space_dummy: var_names += ['s_dummy']


    # prepare data for jpcmci by placing each dataset into a dictionary
    data_dict = {}
    dummy_data_time = np.identity(num_time)
    for i in range(num_domains):
        df = _obs_itv[_obs_itv.index.get_level_values('domain') == i]
        dummy_data_space = np.zeros((num_time, num_domains))
        dummy_data_space[:, i] = 1

        if len(df) > 0: 
            data = df.to_numpy()
            if time_dummy:  data = np.hstack((data, dummy_data_time))
            if space_dummy: data = np.hstack((data, dummy_data_space))
            data_dict.update({i: data})  


    # specify node types: system (endogenous), time_context (time-lagged exogenous), space_context (non-lagged exogenous)
    node_classification = dict(zip(
        _obs_itv.columns,
        ["space_context" if "_itv" in col else "system" for col in _obs_itv.columns],
    ))
    if time_context: node_classification['phase'] = "time_context"

    observed_indices_time = [i for i, col in enumerate(_obs_itv.columns) if node_classification[col] == "time_context"]
    t_context_nodes = list(range(
        num_system_vars, 
        num_system_vars + len(observed_indices_time)
    ))

    observed_indices_space = [i for i, col in enumerate(_obs_itv.columns) if node_classification[col] == "space_context"]
    s_context_nodes = list(range(
        num_system_vars + len(observed_indices_time), 
        num_system_vars + len(observed_indices_time) + len(observed_indices_space)
    ))

    system_indices = [i for i, col in enumerate(_obs_itv.columns) if node_classification[col] == "system"]
    observed_indices = system_indices + observed_indices_time + observed_indices_space

    node_classification_jpcmci = dict(zip(observed_indices, node_classification.values()))
    vector_vars = {i: [(i, 0)] for i in system_indices + t_context_nodes + s_context_nodes}

    new_idx = (num_system_vars + num_context_vars - 1)
    if time_dummy:  
        new_idx += 1
        t_dummy_idx = list(range(new_idx, new_idx + num_time))
        node_classification_jpcmci.update({new_idx : "time_dummy"})
        vector_vars[new_idx] = [(i, 0) for i in t_dummy_idx]
        
    if space_dummy: 
        new_idx += 1
        s_dummy_idx = list(range((data).shape[1] - num_domains, (data).shape[1]))
        node_classification_jpcmci.update({new_idx : "space_dummy"})
        vector_vars[new_idx] = [(i, 0) for i in s_dummy_idx]


    # specify the data_types: 0 is continuous; 1 discrete data
    data_type1 =  np.zeros((num_domains, num_time, num_system_vars), dtype='int')
    data_type2 =  np.ones((num_domains, num_time, num_context_vars), dtype='int')
    if time_dummy: data_type2 = np.concatenate([data_type2, np.ones((num_domains, num_time, num_time), dtype='int'),], axis=2)
    if space_dummy: data_type2 = np.concatenate([data_type2, np.ones((num_domains, num_time, num_domains), dtype='int')], axis=2)
    data_type = np.concatenate([data_type1, data_type2], axis=2)


    # run jpcmciplus
    dataframe = pp.DataFrame(
        data=data_dict,
        analysis_mode='multiple',
        var_names=var_names,
        data_type=data_type,
        vector_vars=vector_vars,
    )
    jpcmciplus = JPCMCIplus(
        dataframe=dataframe, 
        cond_ind_test=ci_test,
        node_classification=node_classification_jpcmci,
        verbosity=0,
    )
    # link assumption 1 = no lag-0 links between endogenous -> endogenous; 
    # link assumption 2 = no lag-0 links between non-corresponding exogenous -> endogenous; 
    results = jpcmciplus.run_jpcmciplus(
        tau_min=tau_min, 
        tau_max=tau_max, 
        pc_alpha=alpha, 
        reset_lagged_links=True,
        link_assumptions=get_link_assumptions(jpcmciplus, tau_min, tau_max, num_system_vars, assumptions=[1,2]), 
    )

    return results, jpcmciplus, dataframe, var_names


