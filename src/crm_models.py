"""CRM models."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skgstat import OrdinaryKriging, Variogram


def calc_rate(times, dp, inj, tau, J, f, q0):
    """CRM equation."""
    nexp = torch.exp(-times / tau)
    pexp = torch.exp(times / tau)
    rates = q0 * nexp

    rates[:, 1:] += ((f/tau)*nexp*torch.cumsum(pexp*inj, dim=1))[:, :-1]*times.diff()
    rates[:, 1:] += -J*nexp[:, 1:]*torch.cumsum(pexp[:, 1:]*dp, dim=1)
    return rates

class CRM(nn.Module):#pylint:disable=too-many-instance-attributes
    """Basic CRM model."""
    def __init__(self, n=1, tau=1, J=1, f=1, tau_min=1e-3, J_min=1e-3, f_min=1e-3):
        super().__init__()
        self._J = nn.Parameter(J * (1 + 0.1*torch.randn(n, 1)))
        self._f = nn.Parameter(f * (1 + 0.1*torch.randn(n, 1)))
        self._tau = nn.Parameter(tau * (1 + 0.1*torch.randn(n, 1)))
        self.n = n
        self.tau_min = tau_min
        self.J_min = J_min
        self.f_min = f_min
        self.tau_ok = None
        self.J_ok = None
        self.f_ok = None
        self.V_tau = None
        self.V_J = None
        self.V_f = None
        self.fitted = False

    def tau(self, locs):
        """Tau field."""
        if self.fitted:
            res = self.tau_ok.transform(locs)
            res[np.isnan(res)] = 0
            return torch.clip(torch.tensor(res), min=self.tau_min).reshape(-1, 1)
        return self.tau_min + F.softplus(self._tau)

    def J(self, locs):
        """J field."""
        if self.fitted:
            res = self.J_ok.transform(locs)
            res[np.isnan(res)] = 0
            return torch.clip(torch.tensor(res), min=self.J_min).reshape(-1, 1)
        return self.J_min + F.softplus(self._J)

    def f(self, locs):
        """f field."""
        if self.fitted:
            res = self.f_ok.transform(locs)
            res[np.isnan(res)] = 0
            return torch.clip(torch.tensor(res), min=self.f_min).reshape(-1, 1)
        return self.f_min + F.softplus(self._f)

    def fit_kriging(self, x, maxlag='mean', n_lags=10, model='spherical', normalize=False,
                    min_points=5, max_points=50):
        """Fit ordinary kriging."""
        self.fitted = False

        y = self.tau(x).detach().ravel()
        self.V_tau = Variogram(x, y, maxlag=maxlag, n_lags=n_lags, model=model, normalize=normalize)
        self.tau_ok = OrdinaryKriging(self.V_tau, min_points=min_points, max_points=max_points, mode='exact')

        y = self.J(x).detach().ravel()
        self.V_J = Variogram(x, y, maxlag=maxlag, n_lags=n_lags, model=model, normalize=normalize)
        self.J_ok = OrdinaryKriging(self.V_J, min_points=min_points, max_points=max_points, mode='exact')

        y = self.f(x).detach().ravel()
        self.V_f = Variogram(x, y, maxlag=maxlag, n_lags=n_lags, model=model, normalize=normalize)
        self.f_ok = OrdinaryKriging(self.V_f, min_points=min_points, max_points=max_points, mode='exact')

        self.fitted = True

    def forward(self, times, locs, dp, inj, q0):
        """Forward run."""
        locs = locs.detach().numpy()
        return calc_rate(times, dp, inj, self.tau(locs), self.J(locs), self.f(locs), q0)

class SpatialCRM(nn.Module):
    """Spatial CRM model."""
    def __init__(self, h=1, tau_min=1e-3, J_min=0, f_min=1e-3):
        super().__init__()
        self._J = nn.Sequential(nn.Linear(2, h),
                                nn.ELU(),
                                nn.Linear(h, 1),
                                nn.Softplus())
        self._f = nn.Sequential(nn.Linear(2, h),
                                nn.ELU(),
                                nn.Linear(h, 1),
                                nn.Softplus())
        self._tau = nn.Sequential(nn.Linear(2, h),
                                  nn.ELU(),
                                  nn.Linear(h, 1),
                                  nn.Softplus())
        self.tau_min = tau_min
        self.J_min = J_min
        self.f_min = f_min

    def tau(self, x):
        """Tau field."""
        return self.tau_min + self._tau(x)

    def J(self, x):
        """J field."""
        return self.J_min + self._J(x)

    def f(self, x):
        """f field."""
        return self.f_min + self._f(x)

    def forward(self, times, locs, dp, inj, q0):
        """Forward run."""
        return calc_rate(times, dp, inj, self.tau(locs), self.J(locs), self.f(locs), q0)
