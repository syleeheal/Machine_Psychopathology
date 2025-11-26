
import warnings
import numpy as np
import scipy.stats as stats
import dcor 

from sklearn.preprocessing import power_transform
from sklearn.neural_network import MLPRegressor

import tigramite
from tigramite.independence_tests.independence_tests_base import CondIndTest

class MLPCI(CondIndTest):

    @property
    def measure(self):
        return self._measure

    def __init__(self, dataset_length, pr_params=None, **kwargs):
        self._measure = 'mlp_ci'
        self.two_sided = False
        self.residual_based = True
        self.eps = 0
        self.dataset_length = dataset_length
        self.seed = 42

        CondIndTest.__init__(self, **kwargs)
                
        self.pr_params = dict(
            hidden_layer_sizes=(8,8,),
            activation='relu',
            solver='lbfgs',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.01,
            early_stopping=True,
            max_iter=200,
            n_iter_no_change=30,
            validation_fraction=0.5,
            batch_size='auto',
            shuffle=True,
            random_state=self.seed,
        )

        if pr_params:
            self.pr_params.update(pr_params)
                    
    @staticmethod
    def _safe_corr(a, b, kind="distance"):
        """Compute correlation between a and b."""        
        r = dcor.distance_correlation(a, b)
        return 0.0 if np.isnan(r) else abs(r)


    def _fit_predict_reg(self, Z, y):
        """Fit PR model of y on Z, return residuals."""

        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        reg = MLPRegressor(**self.pr_params)
        for i in range(Z.shape[1]):
            # if non-binary variable, apply power transform
            if len(np.unique(Z[:, i])) > 2:
                Z[:, i] = power_transform(Z[:, i].reshape(-1, 1), method='yeo-johnson', standardize=True).reshape(-1)

        y_log = np.log1p(y) 
        reg.fit(Z, y_log)
        yhat = reg.predict(Z)
        yhat = np.expm1(yhat)  # inverse of log1p
                
        return y - yhat

    def _get_single_residuals(self, array, target_var):
        """Return residuals of an PR regression of target_var on Z.
        If no Z provided (dim <= 2), return the target series."""
        dim, T = array.shape
        arr = array.astype(np.float64, copy=True)
        y = arr[target_var, :]

        if dim <= 2:
            return y.copy()

        Z = arr[2:, :].T
        return self._fit_predict_reg(Z, y)

    # -------------------- test statistic --------------------

    def get_dependence_measure(self, array, xyz, data_type=None):
        """Compute correlation on residuals"""

        x_res = self._get_single_residuals(array, target_var=0)
        y_res = self._get_single_residuals(array, target_var=1)
        
        corr = self._safe_corr(x_res, y_res)

        return corr
        
    # -------------------- permutation significance --------------------

    def get_shuffle_significance(self, array, xyz, value, return_null_dist=False, data_type=None):
        """Shuffle significance for residual-based statistic.

        We (block-)shuffle one residual series relative to the other, re-evaluating
        the test statistic on the 2×T residual matrix each time

        Parameters
        ----------
        array : ndarray, shape (dim, T)
            Rows are X, Y, (optional) Z..., columns are observations.
        xyz : ndarray, shape (dim,)
            Identifier array (0 for X rows, 1 for Y rows, 2 for Z rows).
            (Unused here; we build a residual-only array internally.)
        value : float
            Observed statistic (unshuffled).
        return_null_dist : bool, default False
            If True, also returns the sampled null distribution.
        data_type : ignored (present for API compatibility)

        Returns
        -------
        pval : float  (or (pval, null_dist) if return_null_dist)
        """

        # Compute residuals once on the original data
        x_res = self._get_single_residuals(array, target_var=0)
        y_res = self._get_single_residuals(array, target_var=1)
        array_resid = np.array([x_res, y_res])
        xyz_resid = np.array([0, 1])
        
        null_dist = self._get_shuffle_dist(
            array=array_resid,
            xyz=xyz_resid,
            dependence_measure=self.get_dependence_measure,
            sig_samples=self.sig_samples,
            sig_blocklength=self.sig_blocklength,
            verbosity=self.verbosity
        )

        # +1 correction (avoids zero p-values; exact under exchangeability)
        ge = (null_dist >= value).sum()
        pval = (ge + 1.0) / (len(null_dist) + 1.0)
        
        if return_null_dist:
            return pval, null_dist
        return pval

