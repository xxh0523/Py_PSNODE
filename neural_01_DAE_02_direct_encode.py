import argparse, datetime, pathlib, os
from ast import parse
from io import UnsupportedOperation
from functools import reduce
from threading import RLock
from matplotlib import markers

from torch.autograd import grad
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.normalization import LayerNorm
from torch.quantization import default_eval_fn
from tqdm.utils import _screen_shape_linux
from math import e, pi, trunc
from neural_dae.neural_base import DAE_Base, DAE_Curves_Sample, DAE_Event
import torch
import torch.nn as nn
from torch.nn.modules import dropout, linear
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from neural_dae import ODE_Curves_Sample, ODE_Event, ODE_Base
from neural_dae import Euler, Midpoint, RK4
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

from utils import Logger, Losses
from neural_01_DAE_01_no_encode import evalute_model, output_training_process

mpl.use('Agg')

# debug
flg_debug = True
data_path = '/home/xiaotannan/pythonPS/00saved_results/samples/generator_epie/1000_gen31_all_4000_limit_samples_gen_0_muststable/'
is_training = True
is_testing = False
# p_model = '/home/xiaotannan/pythonPS/00saved_results/models/neural_dae/neural_gen_0_20220320_4/model_checkpoint.400'
# p_model = '00saved_results/models/neural_dae/test/model_checkpoint.1'
p_model = '00saved_results/models/neural_dae/test'
device_target = 'cpu'
ncols = 80

# pre settings
larger_than = math.pi
learning_rate = 0.005
sch_gamma = 0.7
loss_record_iter = 10
Loss_func = nn.functional.mse_loss # mse is not recommended, because omega is way too small in most of the cases
lamda_x_loss = 1
gradient_clip = 1

# fig set
pic_num = 5
line_width = 1
mark_size = 2


class Init_Func(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(Init_Func, self).__init__()
        self.init_fun = nn.Sequential(#nn.Linear(z_dim+v_dim+i_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                      nn.Linear(z_dim+v_dim+i_dim, hidden_dim), nn.ELU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                      nn.Linear(hidden_dim, x_dim))
    
    def forward(self, z0: torch.Tensor, v0: torch.Tensor, i0: torch.Tensor):
        return self.init_fun(torch.cat([z0, v0, i0], dim=-1))


class DE_Func(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(DE_Func, self).__init__()
        if z_dim == 0:
            self.x_dot = nn.Sequential(#nn.Linear(int(3*3*hidden_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                       nn.Linear(int(3*3*hidden_dim), hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        else:
            self.x_dot = nn.Sequential(#nn.Linear(int(3*4*hidden_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                       nn.Linear(int(3*4*hidden_dim), hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t0: torch.Tensor, xt: torch.Tensor, zt: torch.Tensor, vt: torch.Tensor, it: torch.Tensor, all_initial: torch.Tensor):
        xt_all = torch.cat((xt, zt, vt, it), dim=-1)
        return self.x_dot.forward(torch.cat((all_initial, xt_all-all_initial, xt_all), dim=-1))


class AE_Func(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(AE_Func, self).__init__()
        if z_dim == 0:
            self.i_calculator = nn.Sequential(#nn.Linear(int((3+2)*hidden_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                              nn.Linear(int((3+2)*hidden_dim), hidden_dim), nn.ELU(),
                                              nn.Linear(hidden_dim, hidden_dim))
        else:
            self.i_calculator = nn.Sequential(#nn.Linear(int((4+3)*hidden_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                              nn.Linear(int((4+3)*hidden_dim), hidden_dim), nn.ELU(),
                                              nn.Linear(hidden_dim, hidden_dim))
    
    def forward(self, xt: torch.Tensor, zt: torch.Tensor, vt: torch.Tensor, all_initial: torch.Tensor):
        return self.i_calculator(torch.cat((all_initial, xt, zt, vt), dim=-1))


class DAE_Model(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(DAE_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.x_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, x_dim))
        self.z_encoder = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim)) if z_dim != 0 else None
        self.v_encoder = nn.Sequential(nn.Linear(v_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.i_encoder = nn.Sequential(nn.Linear(i_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.i_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, i_dim))
        self.init_func = Init_Func(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, i_dim=i_dim, hidden_dim=hidden_dim)
        self.de_func = DE_Func(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, i_dim=i_dim, hidden_dim=hidden_dim)
        self.ae_func = AE_Func(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, i_dim=i_dim, hidden_dim=hidden_dim)
        self.solver = Euler()
        self.event = DAE_Event()
        
    def forward(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor, i: torch.Tensor, event_t: torch.Tensor, z_jump: torch.Tensor, v_jump: torch.Tensor):
        x0 = self.init_func(z.permute(1,0,2)[0], v.permute(1,0,2)[0], i.permute(1,0,2)[0])
        Xh0 = self.x_encoder(x0)
        Xh = self.x_encoder(x).permute(1,0,2)
        Zh = z.permute(1,0,2) if self.z_encoder is None else self.z_encoder(z).permute(1,0,2)
        Vh = self.v_encoder(v).permute(1,0,2)
        Ih = self.i_encoder(i).permute(1,0,2)
        all_initial = torch.cat((Xh0, Zh[0], Vh[0], Ih[0]), dim=-1)
        Zh_jump = z_jump if self.z_encoder is None else self.z_encoder(z_jump)
        Vh_jump = self.v_encoder(v_jump)
        self.event.set_event(t=event_t, z=Zh_jump, v=Vh_jump)
        # batch, time, variable -> time, batch, variable
        Xh_solution, Ih_solution = self.solver.integrate_DAE(x_init=Xh0,
                                                             x_func=self.de_func,
                                                             i_func=self.ae_func,
                                                             t=t.permute(1,0,2), 
                                                             x=Xh, 
                                                             z=Zh, 
                                                             v=Vh,
                                                             i=Ih,
                                                             all_initial=all_initial,
                                                             event_fn=self.event.event_fn, 
                                                             jump_change_fn=self.event.jump_change_fn)
        # time, batch, variable -> batch, time, variable
        x_pred = self.x_decoder(Xh_solution)
        x_pred[0] = x0
        x_re = self.x_decoder(Xh)
        i_re = self.i_decoder(Ih)
        return x_pred.permute(1, 0, 2), self.i_decoder(Ih_solution).permute(1, 0, 2), x_re.permute(1, 0, 2), i_re.permute(1, 0, 2)
    
    def save_model(self, path: pathlib.Path):
        if not path.exists(): path.mkdir()
        with open(str(path / 'dim.txt'), 'w') as f:
            f.write(str(self.hidden_dim))
        sm = torch.jit.script(self.x_encoder)
        sm.save(str(path / 'x_encoder.pt'))
        sm = torch.jit.script(self.x_decoder)
        sm.save(str(path / 'x_decoder.pt'))
        if self.z_encoder is not None:
            sm = torch.jit.script(self.z_encoder)
            sm.save(str(path / 'z_encoder.pt'))
        sm = torch.jit.script(self.v_encoder)
        sm.save(str(path / 'v_encoder.pt'))
        sm = torch.jit.script(self.i_encoder)
        sm.save(str(path / 'i_encoder.pt'))
        sm = torch.jit.script(self.i_decoder)
        sm.save(str(path / 'i_decoder.pt'))
        sm = torch.jit.script(self.init_func)
        sm.save(str(path / 'init_func.pt'))
        sm = torch.jit.script(self.de_func)
        sm.save(str(path / 'de_func.pt'))
        sm = torch.jit.script(self.ae_func)
        sm.save(str(path / 'ae_func.pt'))
    
    def final_save(self, path: pathlib.Path):
        if not path.exists(): path.mkdir()
        with open(str(path / 'dim.txt'), 'w') as f:
            f.write(str(self.hidden_dim))
        sm = torch.jit.script(self.x_encoder.to('cpu'))
        sm.save(str(path / 'x_encoder.pt'))
        sm = torch.jit.script(self.x_decoder.to('cpu'))
        sm.save(str(path / 'x_decoder.pt'))
        if self.z_encoder is not None:
            sm = torch.jit.script(self.z_encoder.to('cpu'))
            sm.save(str(path / 'z_encoder.pt'))
        sm = torch.jit.script(self.v_encoder.to('cpu'))
        sm.save(str(path / 'v_encoder.pt'))
        sm = torch.jit.script(self.i_encoder.to('cpu'))
        sm.save(str(path / 'i_encoder.pt'))
        sm = torch.jit.script(self.i_decoder.to('cpu'))
        sm.save(str(path / 'i_decoder.pt'))
        sm = torch.jit.script(self.init_func.to('cpu'))
        sm.save(str(path / 'init_func.pt'))
        sm = torch.jit.script(self.de_func.to('cpu'))
        sm.save(str(path / 'de_func.pt'))
        sm = torch.jit.script(self.ae_func.to('cpu'))
        sm.save(str(path / 'ae_func.pt')) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--device', type=str,
                        help='Choose device to be used, "gpu" or "cpu". Default value is "cpu".',
                        default='cpu')
    parser.add_argument('--id', type=int,
                        help='If using gpu, choose which gpu to be used. Default value is 0.',
                        default=0)
    
    # training, testing, saving, drawing
    parser.add_argument('--training', action='store_true',
                        help='Call training process, --train_data and --test_data needed.',
                        required=False)
    parser.add_argument('--testing', action='store_true',
                        help='Call testing process, --model and --test_data needed.',
                        required=False)
    parser.add_argument('--saving', action='store_true',
                        help='Call saving process, --model needed.',
                        required=False)
    parser.add_argument('--drawing', action='store_true',
                        help='Call drawing pic process, --testing, --model, and --test_data needed.',
                        required=False)
    
    # data and model
    parser.add_argument('--train_data', type=str,
                        help='Training data File Path',
                        required=False, default='./results/samples_neural_gen_2_training.npz')
    parser.add_argument('--test_data', type=str,
                        help='Testing data File Path',
                        required=False, default='./results/samples_neural_gen_2_testing.npz')
    parser.add_argument('--model', type=str,
                        help='<>\tModel dump/load path, directory can be automatically created, but file must exists.',
                        required=False, default='00saved_results/models/neural_dae/test')
    
    # training settings
    parser.add_argument('--num', type=int,
                        help='Set training set size. Default value is 3200.',
                        required=False, default=3200)
    parser.add_argument('--batch', type=int,
                        help='Set mini-batch size. Default value is 64.',
                        required=False, default=64)
    parser.add_argument('--hidden', type=int,
                        help='Set hidden dimentionality. Default value is 128.',
                        required=False, default=128)
    parser.add_argument('--epoch', type=int,
                        help='Set number of training epoch. Default value is 400.',
                        required=False, default=400)
    parser.add_argument('--step', type=int,
                        help='Set length of training series. Default value is 1001.',
                        required=False, default=1001)

    args = parser.parse_args()

    if flg_debug:
        args.training = is_training
        args.testing = is_testing
        args.drawing = True
        args.saving = True
        args.train_data = data_path + '/training.npz'
        args.test_data = data_path + '/testing.npz'
        args.model = p_model
        args.device = device_target
        args.hidden = 64
        args.epoch = 1
        args.num = 100
        args.batch = 2

    # device setting
    if args.device.lower() == 'cpu':
        device = torch.device('cpu')
        print(f'Device is {device}')
    elif args.device.lower() == 'gpu':
        device = torch.device('cuda:' + str(args.id))
        print(f'Device is {device}')
    else:
        raise Exception('Arguments "--device" is illegal. Expected "cpu" or "gpu" but ' + args.device)

    # training or testing
    if args.training:
        assert args.train_data is not None and args.test_data is not None, 'Traning set or testing set missing! Please check.'

        # load data
        training_dataset = DAE_Curves_Sample(data_path=args.train_data, device=device, num_sample=args.num, cut_length=args.step, contain_larger_than=larger_than)
        training_dataloader = DataLoader(training_dataset, batch_size=args.batch, shuffle=True)
        testing_dataset = DAE_Curves_Sample(data_path=args.test_data, device=device, cut_length=args.step)
        testing_dataloader = DataLoader(testing_dataset, batch_size=max(int(testing_dataset.t.shape[0]/10),1), shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=training_dataset.x.shape[-1], z_dim=training_dataset.z.shape[-1],
                                v_dim=training_dataset.v.shape[-1], i_dim=training_dataset.i.shape[-1],
                                hidden_dim=args.hidden).to(device)
        opt_Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt_Adam, step_size=max(int(args.epoch/10),1), gamma=sch_gamma)
        
        # model path
        model_path = pathlib.Path(args.model)
        # if --model and model path is a existing file, meaning continue training
        if not model_path.exists(): model_path.mkdir()
        if model_path.is_dir() is False:
            assert model_path.exists(), f'{model_path} does not exist!'
            model.load_state_dict(torch.load(args.model, map_location=device))
            model_path = model_path.parent / (model_path.name + '_branch')
            if not model_path.exists(): model_path.mkdir()
        
        # loss record
        x_loss_record = np.zeros(loss_record_iter)
        i_loss_record = np.zeros(loss_record_iter)
        loss_record = np.zeros(loss_record_iter)
        # gradient norm
        gradient_record = np.zeros(loss_record_iter)
        # error list
        train_error_list = list()
        eval_error_list = list()
        
        # logger definition
        logger = Logger(model_path, 'training.log', 'testing.log')
        my_loss = Losses(log=logger)
        logger.training_log(f'training_data: {args.train_data}, \
                              testing_data: {args.test_data}, \
                              train_size: {args.num}, \
                              batch_size: {args.batch}, \
                              hidden_dim: {args.hidden}, \
                              epoch: {args.epoch}, \
                              cut_length: {args.step}, \
                              learning_rate: {learning_rate}')

        # initial model testing
        logger.testing_log('======================================================================================')
        logger.testing_log(f'Initial evaluate on training set.')
        if args.drawing: pic_path = pathlib.Path(model_path / 'pics')
        else: pic_path = None
        eval_error_list.append(evalute_model(model=model, Loss_func=Loss_func, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Testing_Epoch_0', pic_path=pic_path, show_larger_than=larger_than))
        logger.testing_log('======================================================================================')

        # start training
        logger.training_log('Start training 2nd-order Neural Generator Model')
        logger.training_log('======================================================================================')
        
        for epoch in tqdm(range(1, args.epoch+1), desc='Epoch', ncols=ncols):
            # set to train
            model.train()
            # if epoch < sch_step: model.solver.flg_input_true_i = True
            # else: model.solver.flg_input_true_i = False
            
            for i_batch, data_batch in enumerate(tqdm(training_dataloader, desc=f'Epoch {epoch} Training', leave=False, ncols=ncols)):
                # transfer to device
                sample_batched = [d.to(device) for d in data_batch]
                # parse
                t, x, z, v, i, event_t, z_jump, v_jump, mask = sample_batched
                
                # forward
                x_pred, i_pred, x_re, i_re = model.forward(t=t, x=x, z=z, v=v, i=i, event_t=event_t, z_jump=z_jump, v_jump=v_jump)
                
                # cal loss
                x_loss = (  torch.sum(Loss_func(x_pred, x, reduction='none') * mask)
                        #   + torch.sum(Loss_func(x_pred[:,:,1:2], x[:,:,1:2], reduction='none') * mask) * 100
                        #   + torch.sum(Loss_func(x_pred[:,:,2:6], x[:,:,2:6], reduction='none') * mask) * 10
                          ) / torch.sum(mask)
                i_loss = torch.sum(Loss_func(i_pred, i, reduction='none') * mask) / torch.sum(mask)
                recon_loss = Loss_func(x_re, x) + Loss_func(i_re, i)
                loss = x_loss + i_loss + Loss_func(x[:, 0, :], x_pred[:, 0, :]) + Loss_func(i[:, 0, :], i_pred[:, 0, :]) + recon_loss

                # backward
                opt_Adam.zero_grad()
                loss.backward()
                opt_Adam.step()

                # grad clip
                total_norm = 0.0
                parameters = list()
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        nn.utils.clip_grad.clip_grad_norm_(p, gradient_clip)
                        parameters.append(p)
                if len(parameters) == 0:
                    total_norm = 0.0
                else:
                    device = parameters[0].grad.device
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 1).to(device) for p in parameters]), 2.0).item()
                gradient_record[i_batch % loss_record_iter] = total_norm

                x_loss_record[i_batch % loss_record_iter] = x_loss.cpu().detach().item()
                i_loss_record[i_batch % loss_record_iter] = i_loss.cpu().detach().item()
                loss_record[i_batch % loss_record_iter] = loss.cpu().detach().item()

                # output loss every 5 iteration
                if (i_batch + 1) % loss_record_iter == 0:
                    e_x = sum(x_loss_record) / loss_record_iter
                    e_i = sum(i_loss_record) / loss_record_iter
                    e_all = sum(loss_record) / loss_record_iter
                    g_n = sum(gradient_record) / loss_record_iter
                    logger.training_log(f'Training epoch {epoch}: Batch{i_batch+1-loss_record_iter:4} to {i_batch+1:4}: x_loss: {e_x:14.10f}, i_loss: {e_i:14.10f}, loss: {e_all:14.10f}, gradient_norm: {g_n:14.10f}.')
                    train_error_list.append([e_x, e_i, e_all])
            logger.training_log('--------------------------------------------------------------------------------------')

            # scheduler for each epoch
            scheduler.step()
            # save model at each epoch
            torch.save(model.state_dict(), model_path / f'model_checkpoint.{epoch}')

            # model testing
            logger.testing_log('======================================================================================')
            logger.testing_log(f'Training Epoch {epoch}, evaluate on training set.')
            if args.drawing: pic_path = pathlib.Path(model_path / 'pics')
            else: pic_path = None
            eval_error_list.append(evalute_model(model=model, Loss_func=Loss_func, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Testing_Epoch_{epoch}', pic_path=pic_path, show_larger_than=larger_than))
            logger.testing_log('======================================================================================')

            # save results
            np.savez(str(model_path / 'train_and_eval.npz'), train=train_error_list, eval=eval_error_list, dtype=object)
            model.save_model(model_path / 'saved model')
        # fin
        model.final_save(model_path / 'saved model')
        output_training_process(logger=logger, eval=eval_error_list)
    elif args.testing:
        assert args.model is not None and args.test_data is not None, 'Model or testing set missing! Pleses check.'
        
        # load data
        testing_dataset = DAE_Curves_Sample(args.test_data, device)
        testing_dataloader = DataLoader(testing_dataset, batch_size=max(int(testing_dataset.t.shape[0]/10),1), shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=testing_dataset.x.shape[-1], z_dim=testing_dataset.z.shape[-1],
                          v_dim=testing_dataset.v.shape[-1], i_dim=testing_dataset.i.shape[-1],
                          hidden_dim=args.hidden).to(device)
        
        # model path
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist!'
        model.load_state_dict(torch.load(args.model, map_location=device))
        if args.drawing: pic_path = model_path.parent / 'pics'
        else: pic_path = None

        # logger definition
        logger = Logger(logfile_path=model_path.parent, test_log_name=f'Model_{model_path.name}_Evaluation.log')
        logger.testing_log(f'Model {model_path} Evaluation')
        logger.testing_log(f'Use testing data: {args.test_data}')
        logger.testing_log('Start evaluating 2nd-order Neural Generator Model')
        logger.testing_log('======================================================================================')

        # evaluate model
        eval_error_list = evalute_model(model=model, Loss_func=Loss_func, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Model {model_path.name} Evaluation', pic_path=pic_path, show_larger_than=larger_than)
        logger.testing_log('======================================================================================')

        # save results
        np.savez(str(model_path.parent / 'evaluation.npz'), train_error_list=list(), eval=eval_error_list, dtype=object)        
    elif args.saving:
        assert args.model is not None and args.test_data is not None, 'Model or testing set missing! Pleses check.'
        
        # load data
        testing_dataset = DAE_Curves_Sample(args.test_data, device)
        testing_dataloader = DataLoader(testing_dataset, batch_size=max(int(testing_dataset.t.shape[0]/10),1), shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=testing_dataset.x.shape[-1], z_dim=testing_dataset.z.shape[-1],
                          v_dim=testing_dataset.v.shape[-1], i_dim=testing_dataset.i.shape[-1],
                          hidden_dim=args.hidden).to(device)
        
        # model path
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist!'
        model.load_state_dict(torch.load(args.model, map_location=device))
        if args.drawing: pic_path = model_path.parent / 'pics'
        else: pic_path = None

        # save model
        model.final_save(model_path.parent / 'saved model')
        print(f'Model {model_path} saved.')       
    else:
        raise Exception('Unknown task. Set "--training" or "--testing".')
