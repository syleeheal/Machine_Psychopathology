import json
import copy

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

import networkx as nx
from networkx.algorithms.community import louvain_communities

from utils import Config, Data_Manager


import scipy.stats as statsc
from sklearn.linear_model import LinearRegression


def bayes_factor_spearman(x, y):
    """
    Compute an approximate Bayes factor BF10 (alternative vs. null) for Spearman correlation.
    Approach: BIC-based approximation with linear regression on rank-transformed variables.
    
    Parameters
    ----------
    x, y : array-like
        Paired observations (same length). NaN-paired rows are removed.

    Returns
    -------
    result : dict
        {
          'BF10': Bayes factor for H1: rho != 0 over H0: rho = 0,
          'rho': Spearman's rho,
          'n': sample size after NaN removal,
          'BIC1': BIC for model with slope (alternative),
          'BIC0': BIC for intercept-only model (null)
        }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    
    # Drop pairs with any NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 paired observations after NaN removal.")
    
    # Spearman correlation (for reporting)
    rho, _ = spearmanr(x, y)
    
    # Rank-transform (average ranks for ties)
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')

    # Design matrices
    X0 = np.column_stack([np.ones(n)])       # intercept-only (null)
    X1 = np.column_stack([np.ones(n), rx])   # intercept + slope (alternative)

    # OLS helper
    def rss(X, yvec):
        beta, *_ = np.linalg.lstsq(X, yvec, rcond=None)
        resid = yvec - X @ beta
        return float(resid @ resid)

    # Residual sums of squares
    RSS0 = rss(X0, ry)
    RSS1 = rss(X1, ry)

    # BICs (k = number of free parameters)
    # BIC = n*ln(RSS/n) + k*ln(n)
    BIC0 = n * np.log(RSS0 / n) + 1 * np.log(n)
    BIC1 = n * np.log(RSS1 / n) + 2 * np.log(n)

    # Bayes factor approximation (Wagenmakers/BIC method):
    # BF10 ≈ exp((BIC0 - BIC1)/2)
    BF10 = np.exp((BIC0 - BIC1) / 2.0)

    return {'BF10': BF10, 'rho': rho, 'n': n, 'BIC1': BIC1, 'BIC0': BIC0}

def parse_preds(series, col_names):
    return pd.DataFrame(series.apply(json.loads).tolist(), columns=col_names)

def set_style():
    sns.set_context('paper', font_scale=1.2, rc={"lines.linewidth": 2.5})
    sns.set_style('whitegrid', {'axes.edgecolor': '0.2', 'grid.linestyle': '-'})

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'TeX Gyre Heros', 'Liberation Sans', 'Arial']

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title

    pd.set_option('future.no_silent_downcasting', True)
    

    c_darkorange_rgb = np.array([249, 97, 0])/255
    c_orange_rgb = np.array([255, 165, 0])/255
    c_blue_rgb = np.array([60,126,176])/255
    c_white_rgb = np.array([255, 255, 255])/255
    c_navy_rgb = np.array([16, 19, 123])/255
    c_gray_rgb = np.array([128, 128, 128])/255
    c_lightgray_rgb = np.array([200, 200, 200])/255

    diverging_cmap = LinearSegmentedColormap.from_list('custom_cmap', [c_blue_rgb, c_white_rgb, c_darkorange_rgb])
    
    return diverging_cmap, c_darkorange_rgb, c_orange_rgb, c_blue_rgb, c_white_rgb, c_navy_rgb, c_gray_rgb, c_lightgray_rgb

def df2obs(out_df, itv_type, itv_thought, sample_id, step, unit, dim_qkey_dict, abbv_dict=None):

    # convert column 'sae_preds' into multiple columns
    if abbv_dict is not None:
        new_cols = list(abbv_dict.values())
    else:
        new_cols = unit
    
    out_df = pd.concat([out_df, parse_preds(out_df['sae_preds'], new_cols)], axis=1)
    out_df = out_df.drop(['sae_preds', 'output_text', 'itv_str', 'itv_str_layer'], axis=1)    
    
    if abbv_dict is not None:
        out_df['itv_thought'] = out_df['itv_thought'].map(abbv_dict).fillna('none')
        
    # aggregate for each sample and step
    obs = out_df.copy()
    obs_by_dim = []
    for key_list in dim_qkey_dict.values():
        _obs = obs[obs['query'].isin(key_list)].drop('query', axis=1)
        _obs = _obs.groupby(['itv_type', 'itv_thought', 'step', 'sample_id']).max() # max pool by question group within each dimension
        obs_by_dim.append(_obs)
    obs = pd.concat(obs_by_dim)
    obs = obs.groupby(['itv_type', 'itv_thought', 'step', 'sample_id']).sum() # sum pool by all dimensions
    
    # filter for specific itv_type, itv_thought, sample_id, and step
    obs = obs[obs.index.get_level_values('itv_type').isin(itv_type)]
    obs = obs[obs.index.get_level_values('itv_thought').isin(itv_thought)]
    obs = obs[obs.index.get_level_values('step').isin(step)]
    obs = obs[obs.index.get_level_values('sample_id').isin(sample_id)]
    
    # convert column 'itv_thought' into multiple columns
    itv = pd.DataFrame(columns=unit + ['none'], data=np.zeros((len(obs), len(unit)+1)))
    itv_vals = obs.reset_index()['itv_thought']
    itv_types = obs.reset_index()['itv_type']
    for i in range(len(itv_vals)): 
        itv.loc[i, itv_vals[i]] = 0 if itv_types[i] == 'phase_4' else 1

    itv = itv.reset_index(drop=True)
    itv = itv.loc[:, (itv != 0).any(axis=0)].fillna(0)
    itv.columns = itv.columns + '_itv'
    if 'none_itv' in itv.columns:
        itv = itv.drop('none_itv', axis=1)
            
    obs_itv = pd.concat([obs.reset_index(), itv], axis=1)
    obs_itv.set_index(['itv_type', 'sample_id', 'itv_thought', 'step'], inplace=True)

    return obs, itv, obs_itv

def get_trend(df, cols, window=5):
    # Mean across columns -> Group by Sample/Step -> Rolling Mean
    trend = df[cols].mean(axis=1).groupby(['sample_id', 'step']).mean().reset_index()
    trend.columns = ['sample_id', 'step', 'score']
    trend['score'] = trend.groupby('sample_id')['score'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return trend

def get_adj_matrix(dag, num_vars):
    adj0 = np.zeros(((num_vars), (num_vars)))
    adj1 = np.zeros(((num_vars), (num_vars)))          
    for i in range((num_vars)):
        for j in range((num_vars)):
            if dag[i, j, 0] == '': adj0[i, j] = 0
            if dag[i, j, 0] == '-->': adj0[i, j] = 1
            if dag[i, j, 0] == '<--': adj0[j, i] = 1
            if dag[i, j, 1] == '': adj1[i, j] = 0
            if dag[i, j, 1] == '-->': adj1[i, j] = 1
            if dag[i, j, 1] == '<--': adj1[j, i] = 1    
    adj = adj0 + adj1
    adj[adj > 1] = 1
    
    return adj, adj0, adj1

def bootstrap_outcomes(dags, link_removal_threshold, num_full_vars):
        
    adj1_list = []
    adj0_list = []
    
    for i in range(len(dags)):
        adj, adj_lag0, adj_lag1 = get_adj_matrix(dags[i], num_full_vars)
        adj1_list.append(adj_lag1)
        adj0_list.append(adj_lag0)
        
    adj_lag1 = np.stack(adj1_list)
    adj_lag1 = adj_lag1.sum(axis=0)
    adj_lag1[adj_lag1 <= link_removal_threshold] = 0
    adj_lag1[adj_lag1 > link_removal_threshold] = 1
    
    adj_lag0 = np.stack(adj0_list)
    adj_lag0 = adj_lag0.sum(axis=0)
    adj_lag0[adj_lag0 <= link_removal_threshold] = 0
    adj_lag0[adj_lag0 > link_removal_threshold] = 1
    
    adj_cat = adj_lag0 + adj_lag1
    adj_cat[adj_cat > 1] = 1
    
    return adj_cat, adj_lag0, adj_lag1

def get_nx_graphs(adj, adj0, adj1, var_names):
    G0 = nx.DiGraph(adj0[:len(var_names), :len(var_names)])
    G1 = nx.DiGraph(adj1[:len(var_names), :len(var_names)])
    G = nx.DiGraph(adj[:len(var_names), :len(var_names)])

    G1.remove_edges_from(nx.selfloop_edges(G1))

    key_idx_dict = {}
    key_idx_dict.update({idx: val for idx, val in enumerate(var_names)})
    edges = G.edges()

    assert nx.is_directed_acyclic_graph(G0)
    
    return G, G0, G1

def graph_stats(G, var_names, print_output=False):

    # analyze G
    _G = G.copy()
    indegree = nx.in_degree_centrality(_G)
    outdegree = nx.out_degree_centrality(_G)
    betweenness = nx.betweenness_centrality(_G)
    closeness = nx.closeness_centrality(_G)
    c_louvain = louvain_communities(_G)
    c_louvain_dict = {}
    for i, community in enumerate(c_louvain):
        for node in community: c_louvain_dict[node] = i
        
    df = pd.DataFrame({
        'in-d': indegree,
        'out-d': outdegree,
        'btw': betweenness,
        'close': closeness,
        'comm_lv': c_louvain_dict,
    })
    df.index = var_names
    if print_output:
        print(df.applymap(lambda x: round(x, 2)), '\n')
    
    return df

def draw_graph(G1, var_names, diverging_cmap, aie_df=None, save_results=False):
    plt.figure(figsize=(9, 9))
    vmin = -0.1
    vmax = 0.1
    
    pos = nx.circular_layout(G1)
    theta = np.pi / 12  # adjust rotation angle (positive = counterclockwise)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    pos = {k: rotation_matrix @ v for k, v in pos.items()}
    
    G1_full = G1.copy()
    # make G1_full a full graph
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i != j:
                G1_full.add_edge(i, j)

    if aie_df is not None:
        edge_scores1 = []
        for src_v, dst_v in G1.edges:
            src_v_name = var_names[src_v]
            dst_v_name = var_names[dst_v]            
            w = aie_df[(aie_df['parent']==src_v_name) & (aie_df['child']==dst_v_name)]['w_signed'].values
            edge_scores1.extend(w)
    else:
        edge_scores1 = [0.05] * G1.number_of_edges()

    node_size = 2500
    nx.draw(G1, pos, arrowsize=0) 
    nx.draw_networkx_nodes(G1, pos, 
        node_size=node_size, 
        node_color='whitesmoke',
        edgecolors='lightgray',
    )
    nx.draw_networkx_labels(G1, pos, 
        labels=dict(zip(range(len(var_names)), var_names)),    
        font_size=14,
        font_weight='bold',
    )
    nx.draw_networkx_edges(G1_full, pos, 
        node_size=node_size,
        width=2, 
        arrowsize=30, 
        connectionstyle='arc3,rad=0.15', 
        edge_color='white',
        label='Edge Weights',
        arrowstyle='simple',
    )
    nx.draw_networkx_edges(G1, pos, 
        node_size=node_size,
        width=2, 
        arrowsize=30, 
        connectionstyle='arc3,rad=0.15', 
        edge_color=edge_scores1,
        edge_cmap=diverging_cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        label='Edge Weights',
        arrowstyle='simple',
    )
    if save_results:
        plt.savefig(f'figures/causal_graph_{cfg.model_id.split("/")[-1]}.svg', format='svg', bbox_inches='tight')
    
    plt.show()

    fig = plt.figure(figsize=(5, 0.5))
    # Create dedicated axis for the colorbar
    cax = fig.add_axes([0.05, 0.4, 0.9, 0.4]) # [left, bottom, width, height]
    sm = matplotlib.cm.ScalarMappable(
        cmap=diverging_cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Average Intervention Effect (AIE)")
    plt.show()

def preprocess_simul_df(df, df_type, abbv_dict):
    
    trust_score_rubric = {'very high': 1, 'high': 0.75, 'moderate': 0.5, 'low': 0.25, 'very low': 0}
    sev_score_rubric = {'negligible': 1, 'mild': (2/3), 'moderate': (1/3), 'severe': 0}
    bias_score_rubric = {'very positive': 1, 'positive': 0.75, 'neutral': 0.5, 'negative': 0.25, 'very negative': 0}
    conf_score_rubric = {'very high': 1, 'high': 0.75, 'moderate': 0.5, 'low': 0.25, 'very low': 0}
    
    if df_type == 'game':
        
        game_df = df.copy()
        # map to abbv_dict
        game_df['itv_thought'] = game_df['itv_thought'].map(abbv_dict)
        game_df['itv_thought'] = game_df['itv_thought'].fillna('None')
        game_df['itv_thought'] = pd.Categorical(game_df['itv_thought'], categories=list(abbv_dict.values())+['None'], ordered=True)
        
        err_df = pd.DataFrame(columns=['error'], index=game_df.index)
        err_df['error'] = 0
        err_df['sample_id'] = game_df['sample_id']
        err_df['itv_thought'] = game_df['itv_thought']
        err_df['itv_str'] = game_df['itv_str']
        err_df['agent-a'] = game_df['agent-a']
        
        outcome_cols = ['agent_a_message', 'agent_b_message', 'trust_evaluation', 'a_strategy', 'b_strategy', 'agent_a_action', 'agent_b_action']
        for col in outcome_cols:
            err_df['error'] += (game_df[col].str.contains('ERROR', case=True, na=False))

        err_df = err_df.groupby(['sample_id', 'itv_thought', 'itv_str', 'agent-a']).sum()

        err_idx = err_df[err_df['error'] > 3].index
        game_df = game_df.set_index(['sample_id', 'itv_thought', 'itv_str', 'agent-a'])
        game_df = game_df.drop(index=err_idx, errors='ignore').reset_index()
        
        print(f'original df length of {df_type}: {len(df)}; dropped {len(err_idx)*5} samples')

        df_score_game = pd.DataFrame()
        df_score_game['bias-trust'] = game_df['trust_evaluation'].map(trust_score_rubric)

        df_score_game['aggression-threat'] = game_df['b_strategy'].apply(lambda x: 1 if ('threat' in x.lower()) else 0)
        df_score_game['deception-deceive'] = game_df['b_strategy'].apply(lambda x: 1 if ('deceive' in x.lower()) else 0)
        df_score_game['avoidance-giveup'] = game_df['b_strategy'].apply(lambda x: 1 if ('give up' in x.lower()) else 0)
        df_score_game['apologetic-self-blame'] = game_df['b_strategy'].apply(lambda x: 1 if ('self-blame' in x.lower()) else 0)
        df_score_game['prosocial-cooperate'] = game_df['b_strategy'].apply(lambda x: 1 if 'cooperate' in x.lower() else 0)

        df_score_game['aggression-raid'] = game_df['agent_b_action'].apply(lambda x: 1 if 'raid' in x.lower() else 0)
        df_score_game['avoidance-no-action'] = game_df['agent_b_action'].apply(lambda x: 1 if 'no action' in x.lower() else 0)
        df_score_game['prosocial-invest'] = game_df['agent_b_action'].apply(lambda x: 1 if 'invest' in x.lower() else 0)
        df_score_game['sample_id'] = game_df['sample_id']
        df_score_game['itv_str'] = game_df['itv_str']
        df_score_game['itv_thought'] = game_df['itv_thought']
        df_score_game['agent-a'] = game_df['agent-a']
        df_score_game['round'] = game_df['round']
        
        
        df_score_game['id'] = df_score_game['sample_id'].astype(str) + '_' + df_score_game['itv_thought'].astype(str) + '_' + df_score_game['itv_str'].astype(str).astype(str) + '_' + df_score_game['agent-a'].astype(str)
        df_score_game = df_score_game.drop(columns=['sample_id', 'itv_thought', 'itv_str', 'agent-a', 'round'])
        df_score_game = df_score_game.groupby(['id']).mean().reset_index()
        df_score_game['sample_id'] = df_score_game['id'].apply(lambda x: x.split('_')[0])
        df_score_game['itv_thought'] = df_score_game['id'].apply(lambda x: x.split('_')[1])
        df_score_game['itv_str'] = df_score_game['id'].apply(lambda x: float(x.split('_')[2]))
        df_score_game['agent-a'] = df_score_game['id'].apply(lambda x: x.split('_')[3]) 
        df_score_game = df_score_game.drop(columns=['id'])
        
        return game_df, df_score_game
    
    elif df_type == 'social':
        social_df = df.copy()

        social_df['itv_thought'] = social_df['itv_thought'].map(abbv_dict)
        social_df['itv_thought'] = social_df['itv_thought'].fillna('None')
        social_df['itv_thought'] = pd.Categorical(social_df['itv_thought'], categories=list(abbv_dict.values())+['None'], ordered=True)
        social_df['decision_action'] = social_df['decision_action'].astype(str)

        err_df = pd.DataFrame(columns=['error'], index=social_df.index)
        err_df['error'] = 0
        err_df['sample_id'] = social_df['sample_id']
        err_df['itv_thought'] = social_df['itv_thought']
        err_df['itv_str'] = social_df['itv_str']
        err_df['agent-a'] = social_df['agent-a']
        
        outcome_cols = ['agent_a_response', 'agent_b_response', 'analysis_severity', 'analysis_capacity', 'decision_action']
        for col in outcome_cols:
            err_df['error'] += (social_df[col].str.contains('ERROR', case=True, na=False))

        err_df = err_df.groupby(['sample_id', 'itv_thought', 'itv_str', 'agent-a']).sum()

        err_idx = err_df[err_df['error'] > 3].index
        social_df = social_df.set_index(['sample_id', 'itv_thought', 'itv_str', 'agent-a'])
        social_df = social_df.drop(index=err_idx, errors='ignore').reset_index()

        print(f'original df length of {df_type}: {len(df)}; dropped {len(err_idx)*5} samples')        

        df_score_social = pd.DataFrame()
        df_score_social['bias-severity'] = social_df['analysis_severity'].map(sev_score_rubric)
        df_score_social['bias-capacity'] = social_df['analysis_capacity'].map(conf_score_rubric)
        
        df_score_social['aggression-criticize'] = social_df['decision_action'].apply(lambda x: 1 if ('criticize' in x.lower()) else 0)
        df_score_social['deception-white-lie'] = social_df['decision_action'].apply(lambda x: 1 if ('lie' in x.lower()) else 0)
        df_score_social['avoidance-avoid'] = social_df['decision_action'].apply(lambda x: 1 if ('avoid' in x.lower()) else 0)
        df_score_social['apologetic-apologize'] = social_df['decision_action'].apply(lambda x: 1 if ('apologize' in x.lower()) else 0)
        df_score_social['prosocial-help'] = social_df['decision_action'].apply(lambda x: 1 if 'help' in x.lower() else 0)
        
        df_score_social['sample_id'] = social_df['sample_id']
        df_score_social['itv_str'] = social_df['itv_str']
        df_score_social['itv_thought'] = social_df['itv_thought']
        df_score_social['agent-a'] = social_df['agent-a']
        df_score_social['round'] = social_df['round']

        df_score_social['id'] = df_score_social['sample_id'].astype(str) + '_' + df_score_social['itv_thought'].astype(str) + '_' + df_score_social['itv_str'].astype(str).astype(str) + '_' + df_score_social['agent-a'].astype(str)
        df_score_social = df_score_social.drop(columns=['sample_id', 'itv_thought', 'itv_str', 'agent-a', 'round'])
        df_score_social = df_score_social.groupby(['id']).mean().reset_index()
        df_score_social['sample_id'] = df_score_social['id'].apply(lambda x: x.split('_')[0])
        df_score_social['itv_thought'] = df_score_social['id'].apply(lambda x: x.split('_')[1])
        df_score_social['itv_str'] = df_score_social['id'].apply(lambda x: float(x.split('_')[2]))
        df_score_social['agent-a'] = df_score_social['id'].apply(lambda x: x.split('_')[3])
        df_score_social = df_score_social.drop(columns=['id'])

        return social_df, df_score_social
    
    # --- Helper Functions ---

def calculate_metric_change(df, y_name, itv_thought, is_bias_col):
    """
    Calculates percent change based on intervention vs. baseline (None).
    """
    base_df = df[df['itv_thought'] == 'None']
    itv_df = df[df['itv_thought'] == itv_thought]

    x = base_df[y_name]
    y = itv_df[y_name]

    # flag for zero baseline
    zero_base = False
    if not is_bias_col:
        if x.mean() == 0.0:
            zero_base = True

    # Apply offset for calculation stability
    x_adj = x.apply(lambda v: v + 0.01)
    y_adj = y.apply(lambda v: v + 0.01)

    if x_adj.mean() != 0:
        percent_change = ((y_adj.mean() - x_adj.mean()) / abs(x_adj.mean()))
    else:
        percent_change = 0.0

    return percent_change * 100, zero_base

def process_dataset(m_id, data_type, target_cols, bias_cols,abbv_dict, symp_keys):
    """
    Loads and processes a specific simulation type (game or social).
    """
    print(f"Processing {data_type} for {m_id}...")
    
    cfg = Config(m_id)
    dm = Data_Manager(cfg)
    
    # Load data
    raw_df = dm.load_output(data_type=data_type)
    
    # Preprocess (assumes preprocess_simul_df is available in scope)
    df_type_key = 'game' if 'game' in data_type else 'social'
    processed_df, df_score = preprocess_simul_df(raw_df, df_type=df_type_key, abbv_dict=abbv_dict)

    score_container = pd.DataFrame()

    for itv_t in symp_keys:
        # Skip if thought not present
        if len(df_score[df_score['itv_thought'] == itv_t]) == 0:
            continue
        
        # Calculate Metrics
        for y_name in target_cols:
            pct_change, zero_base = calculate_metric_change(
                df_score, 
                y_name, 
                itv_t, 
                is_bias_col=(y_name in bias_cols)
            )
            score_container.loc[itv_t, y_name] = pct_change
            score_container.loc[itv_t, y_name + '_zero_base'] = zero_base
            
    return score_container

def plot_simulation_results(score_df, m_id, abbv_dict, PREFIX_ORDER, save = False):
    """
    Generates the main horizontal bar chart comparison.
    """
    # 1. Organize Columns
    cols = sorted(
        [c for c in score_df.columns if not c.endswith('_zero_base')], 
        key=lambda x: (PREFIX_ORDER.index(x.split('-')[0]) if x.split('-')[0] in PREFIX_ORDER else len(PREFIX_ORDER), x)
    )
    
    # Reindex rows based on abbreviation dictionary
    score_df = score_df.reindex(abbv_dict.values())
    
    # Separate values and zero-base flags
    plot_df = score_df[cols]
    zerobase_df = score_df[[c + '_zero_base' for c in cols]]

    # 2. Calculate Y-Positions for Groups (Custom Spacing)
    prefixes = pd.Index(cols).str.split("-", n=1).str[0]
    unique_prefixes = pd.unique(prefixes)
    
    col_to_y = {}
    current_y = 0.0
    GROUP_GAP = 0.1
    
    for p in unique_prefixes:
        group_cols = [c for c in cols if c.startswith(f"{p}-")]
        for c in group_cols:
            col_to_y[c] = current_y
            current_y += 1
        current_y += GROUP_GAP

    # 3. Define Colors
    # Constructing the specific palette from the original code
    base_palette = sns.color_palette("tab20", n_colors=12)
    # Select specific indices based on original logic: [0, 1, 2, 3, 10, 11] then reordered
    sel_c = [base_palette[i] for i in [0, 1, 2, 3, 10, 11]]
    # Reorder: [0,1] + [10,11] + [3] + [2]
    custom_colors = sel_c[:2] + sel_c[-2:] + sel_c[3:4] + sel_c[2:3]
    
    palette_map = dict(zip(unique_prefixes, custom_colors))
    bar_colors = [palette_map[c.split("-", 1)[0]] for c in cols]

    # 4. Plotting
    n_rows = min(len(plot_df), 12)
    fig, axes = plt.subplots(2, 6, figsize=(16, 8), sharex=True)
    axes = axes.ravel()
    yvals = [col_to_y[c] for c in cols]

    for i, ax in enumerate(axes):
        if i < n_rows:
            vals = plot_df.iloc[i].values
            # We need to match zero_base columns to value columns by index
            current_zero_base = zerobase_df.iloc[i].values
            
            ax.barh(yvals, vals, height=1, color=bar_colors, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{plot_df.index[i].capitalize()}", fontweight='bold')
            ax.set_yticks(yvals)
            ax.axvline(0, lw=1, color="k", alpha=0.3)
            ax.yaxis.grid(True, linestyle='-', alpha=0.8)
            ax.set_axisbelow(True)
            
            # Symlog scale setup
            ax.set_xscale('symlog', linthresh=10, base=10)
            ax.set_xlim(-256, 256)
            ax.set_xticks([-100, -10, 0, 10, 100])
            
            # Y-labels only on left-most plots
            if i % 6 == 0:
                ax.set_yticklabels([c.split("-", 1)[1] for c in cols])
            else:
                ax.set_yticklabels([])

            # X-label only on bottom row
            if i >= 6:
                ax.set_xlabel("Percent change from\nthe unintervened LLM")
            else:
                ax.set_xlabel("")

            # Annotations (Values + Daggers)
            eps_offset = 0.2
            for idx, (y0, v) in enumerate(zip(yvals, vals)):
                if abs(v) > 0.01:
                    x_txt = -eps_offset if v >= 0 else eps_offset
                    ha = 'right' if v >= 0 else 'left'
                    
                    # Determine dagger suffix
                    if current_zero_base[idx]:
                        suffix = "†"
                    else: 
                        suffix = ""
                    
                    ax.text(x_txt, y0, f"{v:.0f}{suffix}", va='center', ha=ha, fontsize=11)
        else:
            ax.axis("off")
            
    plt.setp([a.get_xticklabels() for a in axes], rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    
    if save:
        filename = f'figures/simulation_{m_id.split("/")[-1]}.svg'
        plt.savefig(filename, format='svg', bbox_inches='tight')
        print(f"Saved figure to {filename}")
    
    plt.show()
    


def load_and_clean_data(cfg):
    """Loads output data, handles missing responses, and loads existing results."""
    dm = Data_Manager(cfg) # type: ignore
    
    # Load existing results
    result_path = f'{cfg.outcome_dir}/result_summary.json'
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    # Load and patch dataframe
    out_df = dm.load_output(data_type='robust')
    
    # Handle unavailable responses by injecting zero-vectors
    mask_unavailable = out_df['output_text'].str.contains('Response unavailable', na=False)
    if mask_unavailable.any():
        # Use .loc to avoid SettingWithCopyWarning
        out_df.loc[mask_unavailable, 'sae_preds'] = '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
        
        # Check for parsing errors
        for i in out_df[mask_unavailable].index:
             if out_df.loc[i, 'sae_preds'] is None:
                print(f'Warning: Error at index {i}')

    return out_df, results

def get_group_activations(out_df, group_name, itv_types, step, context_params):
    """
    Wrapper for df2obs to fetch activations for a specific group and step.
    """
    # Filter sample IDs based on group name
    if group_name == 'control':
        group_label = 'control group (single activation)'
    else:
        group_label = 'experimental group (joint activation)'
        
    sample_ids = sorted(out_df[out_df['group'] == group_label]['sample_id'].unique())

    # Prepare clean dataframe for df2obs
    clean_df = out_df.drop(columns=['model_id', 'group', 'unique_id'], errors='ignore')

    # Call external df2obs function
    obs_data = df2obs( # type: ignore
        clean_df,
        itv_type=itv_types,
        itv_thought=context_params['symp_keys'],
        sample_id=sample_ids,
        step=step,
        unit=context_params['symp_keys'],
        abbv_dict=context_params['abbv_dict'],
        dim_qkey_dict=context_params['dim_qkey_dict'],
    )[0]
    
    return obs_data

def align_vectors(obs_x, obs_y, itv_filter=None):
    """
    Aligns X (Pre) and Y (Post) observations using pandas merge. 
    """
    # 1. Calculate means across dimensions (axis=1)
    x_flat = obs_x.mean(axis=1).reset_index(name='val_x')
    
    # Filter Y if specific intervention type is requested
    if itv_filter:
        y_subset = obs_y[obs_y.index.get_level_values('itv_type') == itv_filter]
    else:
        y_subset = obs_y
        
    y_flat = y_subset.mean(axis=1).reset_index(name='val_y')

    # 2. Merge on metadata keys to ensure correct alignment
    # Assuming index columns are: ['itv_type', 'itv_thought', 'sample_id', 'step', etc...]
    # We merge on common keys excluding 'step' and 'itv_type' if they differ logically between X and Y
    merge_keys = ['itv_thought', 'sample_id'] 
    
    aligned = pd.merge(x_flat, y_flat, on=merge_keys, how='inner')
    return aligned

def calculate_slope_and_corr(x, y):
    """Calculates Spearman correlation and Linear Regression slope."""
    corr = statsc.spearmanr(x, y, nan_policy='omit').correlation
    
    # Reshape for sklearn
    X_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
    
    reg = LinearRegression(fit_intercept=True).fit(X_reshaped, y_reshaped)
    slope = reg.coef_[0][0]
    
    return slope, corr

def plot_regression(ax, x, y, color, title_prefix=None, label_y=False):
    """Helper to plot regression line and scatter points."""
    sns.regplot(
        x=x, y=y, ax=ax,
        color=color, 
        line_kws={'color': color, 'linewidth': 4, 'linestyle': '-'}, 
        scatter=False, 
        order=1, 
        x_ci='sd'
    )
    sns.scatterplot(x=x, y=y, ax=ax, color=color, s=30, alpha=0.5, marker='X' if title_prefix == 'Control' else 'o')

