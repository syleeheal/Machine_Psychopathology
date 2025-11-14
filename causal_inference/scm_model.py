import pandas as pd
import numpy as np
import copy

# Scikit-learn imports for modeling
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel as W

from sklearn.metrics import r2_score

# from pysr import PySRRegressor 
from tabulate import tabulate

class StructuralEquationModeler:
    """
    A class to handle the fitting, evaluation, and inference of
    structural equation models (SEMs) for time-series data.

    This class orchestrates a leave-one-out cross-validation workflow,
    fits a specified regression model to the data, evaluates its predictive
    performance, and computes the Average Intervention Effect (AIE) to
    quantify causal link strengths.
    """

    def __init__(self, model_name, adj_lag0, adj_lag1, sys_vars, full_vars, symp_label_dict, random_state=42):
        """
        Initializes the StructuralEquationModeler.

        Args:
            model_name (str): The type of model to use. Options: 'linear', 'poly',
                              'gpr', 'mlp', 'symbolic'.
            adj_lag0 (np.ndarray): Adjacency matrix for contemporaneous effects.
            adj_lag1 (np.ndarray): Adjacency matrix for lagged (t-1) effects.
            sys_vars (list): List of endogenous (child) variable names.
            full_vars (list): List of all variable names (endogenous + exogenous).
            symp_label_dict (dict): Mapping from variable names to integer indices.
            random_state (int): Seed for the random number generator.
        """
        self.model_name = model_name
        self.adj_lag0 = adj_lag0
        self.adj_lag1 = adj_lag1
        self.sys_vars = sys_vars
        self.full_vars = full_vars
        self.symp_label_dict = symp_label_dict
        self.random_state = random_state

    def _get_list_of_data(self, data):
        """Splits a DataFrame into a list of smaller DataFrames."""
        data_list = []
        for itv in self.sys_vars:
            data_itv = data[data.index.get_level_values('itv_thought') == itv]
            sample_ids = data_itv.index.get_level_values('sample_id').unique()
            for sample_id in sample_ids:
                data_sample = data_itv[data_itv.index.get_level_values('sample_id') == sample_id]
                data_list.append(data_sample)
        return data_list
    
    def _get_model_instance(self):
        """Returns an unfitted instance of the specified model."""
        if self.model_name == 'linear':
            reg = LinearRegression(fit_intercept=False)
        if self.model_name == 'poly':
            reg = make_pipeline(PolynomialFeatures(degree=2, interaction_only=False, include_bias=True), Ridge(fit_intercept=False, alpha=1))
        if self.model_name == 'gpr':
            reg = GaussianProcessRegressor(kernel=W() + RBF(), alpha=0.0, random_state=self.random_state)
        if self.model_name == 'mlp':
            reg = MLPRegressor(
                    hidden_layer_sizes=(8,8,),
                    activation='relu',
                    solver='lbfgs',
                    alpha=0.01,
                    learning_rate='adaptive',
                    learning_rate_init=0.01,
                    early_stopping=True,
                    max_iter=500,
                    n_iter_no_change=30,
                    validation_fraction=0.2,
                    batch_size='auto',
                    shuffle=True,
                )
            
        if self.model_name == 'symbolic':
            reg = PySRRegressor(
                binary_operators=["+", "*", "min", "max"],
                unary_operators=["identity", "sqrt", "log1p"],
                maxsize=12, niterations=200, population_size=200,
                batching=True, batch_size=2048, parsimony=0.01,
                turbo=True, loss="loss(x, y) = (x - y)^2",
                progress=False, procs=100
            )
        
        return reg
        
    def _prepare_timeseries_io(self, obs_list, var):
        """Prepares input (X) and output (y) arrays for a given variable."""
        i = self.symp_label_dict[var]
        parents_0 = [self.full_vars[j] for j in range(len(self.full_vars)) if self.adj_lag0[j, i] == 1]
        parents_1 = [self.full_vars[j] for j in range(len(self.full_vars)) if self.adj_lag1[j, i] == 1]

        y, x0, x1 = [], [], []
        for _obs in obs_list:
            y.extend(_obs[var].values[1:])
            x1.extend(_obs[parents_1].values[:-1])
            x0.extend(_obs[parents_0].values[1:])
            
        y = np.array(y)
        x0 = np.array(x0) if parents_0 else np.zeros((len(y), 0))
        x1 = np.array(x1) if parents_1 else np.zeros((len(y), 0))

        X = np.concatenate([x0, x1], axis=1) # shape: (num_samples, num_parents)
        
        return X, y, parents_0, parents_1
        
    def fit(self, train_data_list):
        """Fits one structural equation model for each system variable."""
        models = {}
        for var in self.sys_vars:
            X, y = self._prepare_timeseries_io(train_data_list, var)[:2]
            y_log = np.log1p(y)
            model = self._get_model_instance()
            model.fit(X, y_log)
            models[var] = model
        return models

    def evaluate(self, models, test_data_list):
        """Evaluates the fitted models on a test dataset."""
        perf_rows = []
        for var in self.sys_vars:

            X_test, y_test = self._prepare_timeseries_io(test_data_list, var)[:2]
            y_hat = models[var].predict(X_test)
            y_hat = np.expm1(y_hat)
            r2 = r2_score(y_test, y_hat)
            mse_error = np.mean((y_test - y_hat) ** 2)
            mse_error_scaled = mse_error / (np.var(y_test) + 1e-8)

            perf_rows.append({'var': var, 'r2': round(r2, 2), 'mse_error': round(mse_error, 2), 'mse_error_scaled': round(mse_error_scaled, 2)})

        return pd.DataFrame(perf_rows)

    def cross_validate(self, full_data):
        """
        Performs leave-one-out cross-validation over sample IDs.
        """
        sample_ids = full_data.index.get_level_values('sample_id').unique()
        all_perf_dfs = []

        for test_id in sample_ids:
            train_ids = [i for i in sample_ids if i != test_id]

            # Filter data for train and test sets
            _obs_train = full_data[full_data.index.get_level_values('sample_id').isin(train_ids)]
            _obs_test = full_data[full_data.index.get_level_values('sample_id').isin([test_id])]

            _obs_train_list = self._get_list_of_data(_obs_train)
            _obs_test_list = self._get_list_of_data(_obs_test)
            
            # Fit on training data
            print(f"Fitting models for test fold: {test_id}...")
            models = self.fit(_obs_train_list)
            
            # Evaluate on test data
            perf_df = self.evaluate(models, _obs_test_list)
            all_perf_dfs.append(perf_df)

        final_perf_df = pd.concat(all_perf_dfs, axis=0)
        
        print("\n--- Final Cross-Validation Performance (R²) ---")
        print(tabulate(final_perf_df.groupby('var').mean().round(2).T, headers='keys', tablefmt='pretty'))
        return final_perf_df

    def compute_aie(self, data_list, n_trials=10, n_context=5000, **aie_kwargs):
        """
        Computes the Average Intervention Effect (AIE) over multiple trials.
        In each trial, models are refit on the full dataset.
        """
        all_aie_dfs = []
        print(f"\n--- Starting AIE Calculation ({n_trials} trials) ---")
        for i in range(n_trials):
            print(f"AIE Trial {i+1}/{n_trials}...")
            # Refit models on the full data list for each trial
            trial_models = self.fit(data_list)
            
            aie_df = self._compute_aie_single_run(trial_models, data_list, n_context, **aie_kwargs)
            all_aie_dfs.append(aie_df)

        # Average results across all trials
        aie_dfs_concat = pd.concat(all_aie_dfs, axis=0)
        final_aie_df = aie_dfs_concat.groupby(['child', 'parent', 'lag']).mean().reset_index()
        return final_aie_df

    def _compute_aie_single_run(self, models, data_list, n_context, **aie_kwargs):
        """Wrapper to compute AIE for all variables for a single set of models."""
        all_rows = []
        for var in self.sys_vars:
            if var not in models:
                continue
            df_var = self._compute_aie_for_child(
                var, models[var], data_list, n_context, **aie_kwargs
            )
            all_rows.append(df_var)
        return pd.concat(all_rows, axis=0).reset_index(drop=True)

    def _compute_aie_for_child(self, var, model, data_list, n_context, **kwargs):
        """Computes AIE for edges into a single child variable."""
        q_low = kwargs.get('q_low', 0.1)
        q_high = kwargs.get('q_high', 0.9)
        q_len = kwargs.get('q_len', 0.1)
        per_child_normalize = kwargs.get('per_child_normalize', True)

        rng = np.random.default_rng(self.random_state)
        
        X_context, _, parents_0, parents_1 = self._prepare_timeseries_io(data_list, var)
        
        if X_context.shape[0] < n_context:
            n_context = X_context.shape[0]
            print(f"Warning: n_context ({kwargs.get('n_context', 5000)}) > available data ({X_context.shape[0]}). Using {n_context} samples.")
        
        if n_context == 0:
             return pd.DataFrame()

        X_sample = X_context[rng.choice(X_context.shape[0], size=n_context, replace=False)]

        idx_map = {}
        for col, p in enumerate(parents_0): idx_map[col] = (p, 0)
        for col, p in enumerate(parents_1): idx_map[col + len(parents_0)] = (p, 1)

        yhat = model.predict(X_sample)
        yhat = np.expm1(yhat)
        f_range = np.maximum(np.ptp(yhat), 1e-8)
        
        rows = []
        for col_idx, (parent, lag) in idx_map.items():
            pairs = list(zip(np.arange(q_low, q_high, q_len), np.arange(q_low + q_len, q_high + q_len, q_len)))
            
            X_context_col = X_context[:, col_idx]
            X_min = np.min(X_context_col)
            X_max = np.max(X_context_col)
            
            deltas = []
            for ql, qh in pairs:
                X_lo, X_hi = X_sample.copy(), X_sample.copy()
                X_lo[:, col_idx], X_hi[:, col_idx] = ql, qh
                y_hi = model.predict(X_hi) 
                y_lo = model.predict(X_lo) 
                y_hi = np.expm1(y_hi)
                y_lo = np.expm1(y_lo)
                delta = y_hi - y_lo
                deltas.append(delta)
            
            w_raw_mag = float(np.mean([np.mean(np.abs(d)) for d in deltas]))
            w_raw_signed = float(np.mean([np.mean(d) for d in deltas]))
            rows.append({"child": var, "parent": parent, "lag": lag, "w_raw_mag": w_raw_mag, "sign": np.sign(w_raw_signed)})

        df = pd.DataFrame(rows)
        if not df.empty:
            denom = np.maximum(df["w_raw_mag"].abs().sum(), 1e-12)
            if per_child_normalize:
                df["w_norm"] = df["w_raw_mag"].abs() / denom
            else:
                df["w_norm"] = df["w_raw_mag"] / f_range
        return df
