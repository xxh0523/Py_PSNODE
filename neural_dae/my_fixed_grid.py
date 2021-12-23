###############################################
# The code is developed based on torchdiffeq
# https://github.com/rtqichen/torchdiffeq
###############################################

import torch
from .my_solvers import FixedGridODESolver
import torch.nn as nn
# import torchdiffeq._impl.fixed_grid


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None):
        if v0 is None: f0 = func(t0=t0, xt=x0, zt=z0, all_initial=all_initial)
        else: f0 = func(t0=t0, xt=x0, zt=z0, vt=v0, it=i0, all_initial=all_initial)
        return dt * f0, f0

class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None):
        half_dt = 0.5 * dt
        if v0 is None: 
            f0 = func(t0=t0, xt=x0, zt=z0, all_initial=all_initial)
            x_mid = x0 + f0 * half_dt
            return dt * func(t0=t0+half_dt, xt=x_mid, zt=z0, all_initial=all_initial), f0
        else: 
            f0 = func(t0=t0, xt=x0, zt=z0, vt=v0, it=i0, all_initial=all_initial)
            x_mid = x0 + f0 * half_dt
            return dt * func(t0=t0+half_dt, xt=x_mid, zt=z0, vt=v0, it=i0, all_initial=all_initial), f0


class RK4(FixedGridODESolver):
    order = 4

    def rk4_alt_step_func(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None, f0=None, perturb=False):
        """Smaller error with slightly more compute."""
        k1 = f0
        if v0 is None:
            if k1 is None: k1 = func(t0=t0, xt=x0, zt=z0, all_initial=all_initial)
            k2 = func(t0=t0+dt*_one_third, xt=x0+dt*k1*_one_third, zt=z0, all_initial=all_initial)
            k3 = func(t0=t0+dt*_two_thirds, xt=x0+dt*(k2-k1*_one_third), zt=z0, all_initial=all_initial)
            k4 = func(t0=t1, xt=x0+dt*(k1-k2+k3), zt=z0, all_initial=all_initial)
        else:
            if k1 is None: k1 = func(t0=t0, xt=x0, zt=z0, vt=v0, it=i0, all_initial=all_initial)
            k2 = func(t0=t0+dt*_one_third, xt=x0+dt*k1*_one_third, zt=z0, vt=v0, it=i0, all_initial=all_initial)
            k3 = func(t0=t0+dt*_two_thirds, xt=x0+dt*(k2-k1*_one_third), zt=z0, vt=v0, it=i0, all_initial=all_initial)
            k4 = func(t0=t1, xt=x0+dt*(k1-k2+k3), zt=z0, vt=v0, it=i0, all_initial=all_initial)
        return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

    def _step_func(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None):
        if v0 is None: 
            f0 = func(t0=t0, xt=x0, zt=z0, all_initial=all_initial)
            return self.rk4_alt_step_func(func=func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, all_initial=all_initial, f0=f0), f0
        else: 
            f0 = func(t0=z0, xt=x0, zt=z0, vt=v0, it=i0, all_initial=all_initial)
            return self.rk4_alt_step_func(func=func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, v0=v0, i0=i0, all_initial=all_initial, f0=f0), f0
        
