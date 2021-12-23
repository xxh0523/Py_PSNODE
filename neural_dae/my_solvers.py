###############################################
# The code is developed based on torchdiffeq
# https://github.com/rtqichen/torchdiffeq
###############################################

import abc
from os import TMP_MAX
import torch
import torch.nn as nn
# from TorchDiffEqPack import odesolve_adjoint_sym12

class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, step_size=None, grid_constructor=None, interp="linear"):

        self.step_size = step_size
        self.interp = interp

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda func, x0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None):
        pass
    
    def step_integrate(self, func: nn.Module, t0: torch.Tensor, dt: torch.Tensor, t1: torch.Tensor, x0: torch.Tensor, z0=None, v0=None, i0=None, all_initial=None):
        dx, f0 = self._step_func(func=func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, v0=v0, i0=i0, all_initial=all_initial)
        return x0 + dx, f0
    
    def integrate_ODE(self, x_func: nn.Module, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, all_initial: torch.Tensor, event_fn=None, jump_change_fn=None):
        assert torch.is_tensor(t) and torch.is_tensor(x) and torch.is_tensor(z), 't or x0 or y is not tensor!'
        time_grid = self.grid_constructor(x_func, x, t)
        assert (not torch.any(time_grid[0] != t[0])) and (not torch.any(time_grid[-1] != t[-1])), 'Time grid creation failed!'
        assert (event_fn is None and jump_change_fn is None) or (event_fn is not None and jump_change_fn is not None), 'Event funtion and jump change funtion do not match!'
        assert t.device == x.device == z.device, 't, x0, and y are on different devices!'
        assert time_grid.shape[0] == z.shape[0], 'Dimensions of t and z do not match!'

        x0 = x[0]

        x_solution = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        x_solution[0] = x0

        j = 1
        for t0, t1, z0 in zip(time_grid[:-1], time_grid[1:], z[:-1]):
            dt = t1 - t0
            # support simple event like y jump change
            # TODO complex changes to x, need to rewrite backward function using adjoint sensitivity method
            if event_fn is not None and event_fn(t0) == True:
                z0_jump = jump_change_fn(t0, z0)
                x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0_jump, all_initial=all_initial)
            else:
                x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, all_initial=all_initial)

            x_solution[j] = x1
            x0 = x1
            j += 1

        return x_solution

    def integrate_DAE(self, x_init: torch.Tensor, x_func: nn.Module, i_func: nn.Module, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor, i: torch.Tensor, all_initial: torch.Tensor, event_fn=None, jump_change_fn=None, input_true_x=False, input_true_i=False):
        assert torch.is_tensor(t) and torch.is_tensor(x) and torch.is_tensor(z) and torch.is_tensor(v) and torch.is_tensor(i), 't or x0 or y is not tensor!'
        time_grid = self.grid_constructor(x_func, x, t)
        assert (not torch.any(time_grid[0] != t[0])) and (not torch.any(time_grid[-1] != t[-1])), 'Time grid creation failed!'
        assert (event_fn is None and jump_change_fn is None) or (event_fn is not None and jump_change_fn is not None), 'Event funtion and jump change funtion do not match!'
        assert t.device == x.device == z.device == v.device == i.device, 't, x0, z1, y0, and z2 are on different devices!'
        assert time_grid.shape[0] == z.shape[0] == v.shape[0] == i.shape[0], 'Dimensions of t, z, v, and i do not match!'

        # initial value
        x0 = x_init
        i0 = i_func(xt=x[0], zt=z[0], vt=v[0], all_initial=all_initial) if input_true_x else i_func(xt=x0, zt=z[0], vt=v[0], all_initial=all_initial)

        x_solution = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        x_solution[0] = x0
        i_solution = torch.zeros(i.shape, dtype=i.dtype, device=i.device)
        i_solution[0] = i0

        j = 1
        for t0, t1, z0, z1, v0, v1 in zip(time_grid[:-1], time_grid[1:], z[:-1], z[1:], v[:-1], v[1:]):
            dt = t1 - t0
            # support simple event like z1 and z2 jump change, resulting in jump change of y
            # TODO complex changes to x, need to rewrite backward function using adjoint sensitivity method
            if event_fn is not None and event_fn(t0) == True:
                z0_jump, v0_jump = jump_change_fn(t0, z0, v0)
                i0 = i_func(xt=x0, zt=z0_jump, vt=v0_jump, all_initial=all_initial)
                if input_true_i: x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0_jump, v0=v0_jump, i0=i[j], all_initial=all_initial)
                else: x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0_jump, v0=v0_jump, i0=i0, all_initial=all_initial)
            else:
                if input_true_i: x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, v0=v0, i0=i[j-1], all_initial=all_initial)
                else: x1, _ = self.step_integrate(func=x_func, t0=t0, dt=dt, t1=t1, x0=x0, z0=z0, v0=v0, i0=i0, all_initial=all_initial)

            i1 = i_func(xt=x[j], zt=z1, vt=v1, all_initial=all_initial) if input_true_x else i_func(xt=x1, zt=z1, vt=v1, all_initial=all_initial)
            
            x_solution[j] = x1
            i_solution[j] = i1
            
            x0 = x1
            i0 = i1
            
            j += 1

        return x_solution, i_solution
    
    # def integrate_DAE(self, x_func, t, x, z1, y_func, y, z2, event_fn=None, jump_change_fn=None, encode_x=False, input_true_y=False, input_true_x=False, cat_dim=-1):
    #     assert torch.is_tensor(t) and torch.is_tensor(x) and torch.is_tensor(z1) and torch.is_tensor(y) and torch.is_tensor(z2), 't or x0 or y is not tensor!'
    #     time_grid = self.grid_constructor(x_func, x, t)
    #     assert (not torch.any(time_grid[0] != t[0])) and (not torch.any(time_grid[-1] != t[-1])), 'Time grid creation failed!'
    #     assert (event_fn is None and jump_change_fn is None) or (event_fn is not None and jump_change_fn is not None), 'Event funtion and jump change funtion do not match!'
    #     assert t.device == x.device == z1.device == y.device == z2.device, 't, x0, z1, y0, and z2 are on different devices!'
    #     assert time_grid.shape[0] == z1.shape[0] == z2.shape[0], 'Dimensions of t, z1, and z2 do not match!'

    #     # initial value
    #     x0 = x_func.set_initial(x[0], torch.cat((z1[0], y[0], z2[0]), dim=cat_dim))
    #     y0 = y_func(x0, z2[0])

    #     x_solution = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    #     x_solution[0] = x_func.get_decode_x(x0) if encode_x else x0
    #     y_solution = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
    #     y_solution[0] = y0
    #     if input_true_y: y0 = y[0]

    #     j = 1
    #     for t0, t1, z10, z20, z21 in zip(time_grid[:-1], time_grid[1:], z1[:-1], z2[:-1], z2[1:]):
    #         dt = t1 - t0
    #         # support simple event like z1 and z2 jump change, resulting in jump change of y
    #         # TODO complex changes to x, need to rewrite backward function using adjoint sensitivity method
    #         if event_fn is not None and event_fn(t0) == True:
    #             z10_jump, z20_jump = jump_change_fn(t0, z10, z20)
    #             # y0 = y_func(x_func.get_decode_x(x0), z20_jump) if encode_x else y_func(x0, z20_jump)
    #             y0 = y_func(x0, z20_jump)
    #             x1, _ = self.step_integrate(x_func, t0, dt, t1, x0, torch.cat((z10_jump, y0, z20_jump), dim=cat_dim))
    #         else:
    #             x1, _ = self.step_integrate(x_func, t0, dt, t1, x0, torch.cat((z10, y0, z20), dim=cat_dim))

    #         # y1 = y_func(x_func.get_decode_x(x1), z21) if encode_x else y_func(x1, z21)
    #         y1 = y_func(x[j], z21) if input_true_x else y_func(x1, z21)
            
    #         x_solution[j] = x_func.get_decode_x(x1) if encode_x else x1
    #         y_solution[j] = y1
            
    #         x0 = x1
    #         y0 = y[j] if input_true_y else y1
            
    #         j += 1

    #     return x_solution, y_solution

    def _cubic_hermite_interp(self, t0, x0, f0, t1, x1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * x0 + h10 * dt * f0 + h01 * x1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, x0, x1, t):
        if t == t0:
            return x0
        if t == t1:
            return x1
        slope = (t - t0) / (t1 - t0)
        return x0 + slope * (x1 - x0)
