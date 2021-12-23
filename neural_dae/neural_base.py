from neural_dae.my_solvers import FixedGridODESolver
from random import sample
import numpy as np
from ray.worker import init
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ODE_Curves_Sample(Dataset):
    def __init__(self, data_path, device, num_sample=None):
        super().__init__()
        sample_file = np.load(data_path, allow_pickle=True)
        total_num = sample_file['t'].shape[0]
        index = np.arange(total_num)
        index = np.random.default_rng().choice(index, num_sample) if num_sample is not None else index
        self.data_name = sample_file['name']
        self.t = torch.from_numpy(sample_file['t'][index]) # num_sample * t_dim * 1
        self.x = torch.from_numpy(sample_file['x'][index]) # num_sample * t_dim * x_dim
        self.z = torch.from_numpy(sample_file['z'][index]) # num_sample * t_dim * y_dim
        self.event_t = torch.from_numpy(sample_file['event_t'][index])#.to(device)
        self.z_jump = torch.from_numpy(sample_file['z_jump'][index])#.to(device)
        self.mask = torch.from_numpy(sample_file['mask'][index])#.to(device)
        for tt, xx, zz in zip(self.t, self.x, self.z):
            assert tt.shape[0] == xx.shape[0] == zz.shape[0], 'Sample shapes are wrong!'

    def __len__(self):
        return self.t.shape[0]
    
    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.z[idx], self.event_t[idx], self.z_jump[idx], self.mask[idx]


class ODE_Event():
    def __init__(self):
        self.event_t = None
        self.z_jump = None
    
    def set_event(self, t: torch.Tensor, z: torch.Tensor):
        self.event_t = t
        self.z_jump = z
    
    def event_fn(self, t0: torch.Tensor):
        if self.event_t is None: return False
        if t0[0] in self.event_t[0]: return True
        # for tt, tt0 in zip(self.event_t, t0):
            # if tt0 in tt: return True
        return False
    
    def jump_change_fn(self, t0: torch.Tensor, z0: torch.Tensor):
        z0_jump = z0.clone().detach()
        z0_jump[:] = self.z_jump[:, (self.event_t[0] == t0[0][0]).view(-1)].view(z0_jump.shape)        
        # for i in range(len(self.event_t)):
            # if t0[i] in self.event_t[i]:
                # y0_jump[i] = self.y_jump[i][self.event_t[i] == t0[i]]
        return z0_jump


class DE_Func(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super(DE_Func, self).__init__()
        self.x_encoder = nn.ModuleList()
        self.x_decoder = nn.ModuleList()
        self.Xh_Ext_H = nn.ModuleList()
        self.Xh_dot_H = nn.ModuleList()
        for _ in range(x_dim): 
            self.x_encoder.append(nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(),
                                                nn.Linear(hidden_dim, hidden_dim)))
            self.x_decoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                                nn.Linear(hidden_dim, 1)))
            self.Xh_Ext_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                               nn.Linear(hidden_dim, hidden_dim)))
            self.Xh_dot_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                               nn.Linear(hidden_dim, hidden_dim)))
        self.z_encoder = nn.ModuleList()
        self.Zh_Ext_H = nn.ModuleList()
        for _ in range(z_dim):
            self.z_encoder.append(nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(),
                                                nn.Linear(hidden_dim, hidden_dim)))
            self.Zh_Ext_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                               nn.Linear(hidden_dim, hidden_dim)))
        self.Xh_dot_V = nn.Sequential(nn.Linear(int((x_dim+z_dim)*3), hidden_dim), nn.ELU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                      nn.Linear(hidden_dim, x_dim))
    
    def set_initial(self, x0: torch.Tensor, z0: torch.Tensor):
        self.Xh0 = self.get_encode_Xh(x0, z0)
        return self.Xh0

    def get_encode_Xh(self, x: torch.Tensor, z: torch.Tensor):
        Xh0 = torch.cat([self.x_encoder[i](x[:, i:i+1]) for i in range(x.shape[1])], dim=-2)
        f_Xh0_H = torch.cat([self.Xh_Ext_H[i](Xh0[:, i:i+1]) for i in range(Xh0.shape[1])], dim=-2)
        f_Zh0_H = torch.cat([self.Zh_Ext_H[i](self.z_encoder[i](z[:, i:i+1])) for i in range(z.shape[1])], dim=-2)
        self.f_XZh0_H = torch.cat((f_Xh0_H, f_Zh0_H), dim=-2)
        return Xh0
    
    def get_decode_x(self, Xh: torch.Tensor):
        return torch.cat([self.x_decoder[i](Xh[:, i:i+1]) for i in range(Xh.shape[1])], dim=-2)

    def forward(self, t0: torch.Tensor, Xht: torch.Tensor, zt: torch.Tensor):
        f_Xht_H = torch.cat([self.Xh_Ext_H[i](Xht[:, i:i+1]) for i in range(Xht.shape[1])], dim=-2)
        f_Zht_H = torch.cat([self.Zh_Ext_H[i](self.z_encoder[i](zt[:, i:i+1])) for i in range(zt.shape[1])], dim=-2)
        f_XZht_H = torch.cat((f_Xht_H, f_Zht_H), dim=-2)
        Xht_dot_V = self.Xh_dot_V(torch.cat((f_XZht_H, self.f_XZh0_H, f_XZht_H-self.f_XZh0_H), dim=-2).permute(0, 2, 1)).permute(0, 2, 1)
        return torch.cat([self.Xh_dot_H[i](Xht_dot_V[:, i:i+1]) for i in range(Xht_dot_V.shape[1])], dim=-2)


class ODE_Base(nn.Module):
    def __init__(self, de_func: DE_Func, solver: FixedGridODESolver, flg_encode_x=False):
        super(ODE_Base, self).__init__()
        self.de_function = de_func
        self.solver = solver

    def forward(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, all_initial: torch.Tensor, event_fn=None, jump_change_fn=None):
        out = self.solver.integrate_ODE(x_func=self.de_function, 
                                        t=t, 
                                        x=x, 
                                        z=z, 
                                        all_initial=all_initial,
                                        event_fn=event_fn, 
                                        jump_change_fn=jump_change_fn,
                                        )
        return out 


class DAE_Curves_Sample(Dataset):
    def __init__(self, data_path, device, num_sample=None):
        super(DAE_Curves_Sample, self).__init__()
        sample_file = np.load(data_path, allow_pickle=True)
        total_num = sample_file['t'].shape[0]
        index = np.arange(total_num)
        index = np.random.default_rng().choice(index, num_sample) if num_sample is not None else index
        self.data_name = sample_file['name']
        self.t = torch.from_numpy(sample_file['t'][index])#.to(device)
        self.x = torch.from_numpy(sample_file['x'][index])#.to(device)
        self.z = torch.from_numpy(sample_file['z'][index])#.to(device)
        self.v = torch.from_numpy(sample_file['v'][index])#.to(device)
        self.i = torch.from_numpy(sample_file['i'][index])#.to(device)
        self.event_t = torch.from_numpy(sample_file['event_t'][index])#.to(device)
        self.z_jump = torch.from_numpy(sample_file['z_jump'][index])#.to(device)
        self.v_jump = torch.from_numpy(sample_file['v_jump'][index])#.to(device)
        self.mask = torch.from_numpy(sample_file['mask'][index])#.to(device)
        for tt, xx, zz, vv, ii in zip(self.t, self.x, self.z, self.v, self.i):
            assert tt.shape[0] == xx.shape[0] == zz.shape[0] == vv.shape[0] == ii.shape[0], 'Sample shapes are wrong!'

    def __len__(self):
        return self.t.shape[0]
    
    def __getitem__(self, idx):
        return self.t[idx], self.x[idx], self.z[idx], self.v[idx], self.i[idx], self.event_t[idx], self.z_jump[idx], self.v_jump[idx], self.mask[idx]


class DAE_Event():
    def __init__(self):
        self.event_t = None
        self.z_jump = None
        self.v_jump = None
    
    def set_event(self, t: torch.Tensor, z: torch.Tensor, v: torch.Tensor):
        self.event_t = t
        self.z_jump = z
        self.v_jump = v
    
    def event_fn(self, t0):
        if self.event_t is None: return False
        if t0[0] in self.event_t[0]: return True
        # for tt, tt0 in zip(self.event_t, t0):
            # if tt0 in tt: return True
        return False
    
    def jump_change_fn(self, t0, z0, v0):
        z0_jump = z0.clone().detach()
        v0_jump = v0.clone().detach()
        z0_jump[:] = self.z_jump[:, (self.event_t[0] == t0[0][0]).view(-1)].view(z0_jump.shape)
        v0_jump[:] = self.v_jump[:, (self.event_t[0] == t0[0][0]).view(-1)].view(v0_jump.shape)
        # for i in range(len(self.event_t)):
        #     if t0[i][0] in self.event_t[i]:
        #         z10_jump[i] = self.z1_jump[i][(self.event_t[i] == t0[i][0]).view(-1)]
        #         z20_jump[i] = self.z2_jump[i][(self.event_t[i] == t0[i][0]).view(-1)]
        return z0_jump, v0_jump


class AE_Func(nn.Module):
    def __init__(self, x_dim, v_dim, i_dim, hidden_dim):
        super(AE_Func, self).__init__()
        self.Xh_Ext_H = nn.ModuleList()
        for _ in range(x_dim): 
            self.Xh_Ext_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                               nn.Linear(hidden_dim, hidden_dim)))
        self.z2_encoder = nn.ModuleList()
        self.Z2h_Ext_H = nn.ModuleList()
        for _ in range(i_dim):
            self.z2_encoder.append(nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(),
                                                 nn.Linear(hidden_dim, hidden_dim)))
            self.Z2h_Ext_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                                nn.Linear(hidden_dim, hidden_dim)))
        self.Yh_func_V = nn.Sequential(nn.Linear(int(x_dim+v_dim), hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, i_dim))
        self.y_decoder = nn.ModuleList()
        self.Yh_Ext_H = nn.ModuleList()
        for _ in range(i_dim):
            self.y_decoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                                nn.Linear(hidden_dim, 1)))
            self.Yh_Ext_H.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                               nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, Xht: torch.Tensor, vt: torch.Tensor):
        f_Xht_H = torch.cat([self.Xh_Ext_H[i](Xht[:, i:i+1]) for i in range(Xht.shape[1])], dim=-2) 
        f_Z2ht_H = torch.cat([self.Z2h_Ext_H[i](self.z2_encoder[i](vt[:, i:i+1])) for i in range(vt.shape[1])], dim=-2)
        Yht = self.Yh_func_V(torch.cat((f_Xht_H, f_Z2ht_H), dim=-2).permute(0, 2, 1)).permute(0, 2, 1)
        return torch.cat([self.y_decoder[i](self.Yh_Ext_H[i](Yht[:, i:i+1])) for i in range(Yht.shape[1])], dim=-2)


class DAE_Base(nn.Module):
    def __init__(self, de_func: DE_Func, ae_func: AE_Func, solver: FixedGridODESolver, flg_encode_x=False, flg_input_true_x=False, flg_input_true_i=False):
        super(DAE_Base, self).__init__()
        self.de_function = de_func
        self.ae_function = ae_func
        self.solver = solver
        self.flg_encode_x = flg_encode_x
        self.flg_input_true_x = flg_input_true_x
        self.flg_input_true_i = flg_input_true_i

    def forward(self, t, x, z, v, i, event_fn=None, jump_change_fn=None):
        x_out, i_out = self.solver.integrate_DAE(x_func=self.de_function, 
                                                 i_func=self.ae_function, 
                                                 t=t, 
                                                 x=x, 
                                                 z=z, 
                                                 v=v, 
                                                 i=i, 
                                                 event_fn=event_fn, 
                                                 jump_change_fn=jump_change_fn,
                                                 encode_x=self.flg_encode_x,
                                                 input_true_x=self.flg_input_true_x,
                                                 input_true_i=self.flg_input_true_i)
        return x_out, i_out

