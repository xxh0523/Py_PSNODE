import argparse, datetime, pathlib, os
from io import UnsupportedOperation
from functools import reduce
from threading import RLock
from matplotlib import markers

from torch.autograd import grad
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.normalization import LayerNorm
from torch.quantization import default_eval_fn
from tqdm.utils import _screen_shape_linux
from utils import Logger
from math import e, pi, trunc
from neural_dae.neural_base import AE_Func, DAE_Base, DAE_Curves_Sample, DAE_Event
import torch
import torch.nn as nn
from torch.nn.modules import dropout, linear
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from neural_dae import ODE_Curves_Sample, ODE_Event, DE_Func, ODE_Base
from neural_dae import Euler, Midpoint, RK4
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


# debug
flg_debug = True
data_path = '/home/xiaotannan/pythonPS/00saved_results/samples/generator_epie/300_gen31_all_4000_nolimit_samples_gen_0'
is_training = True
is_testing = False
# p_model = '/home/xiaotannan/pythonPS/00saved_results/models/neural_dae/neural_gen_0_20211128_4/model_checkpoint.400'
p_model = '00saved_results/models/neural_dae/test'
device_target = 'cpu'
ncols = 80

# pre settings
training_sample_num = 3200
batch_size = 64
testing_sample_num = 800
hidden_dim = 64
learning_rate = 0.005
sch_gamma = 0.7
sch_step = 40
num_epoch = 400
loss_record_iter = 10
Loss_func = nn.functional.mse_loss # mse is not recommended, because omega is way too small in most of the cases
lamda_x_loss = 1
gradient_clip = 10

# fig set
pic_num = 2
line_width = 1
mark_size = 2


class DE_Func(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(DE_Func, self).__init__()
        if z_dim == 0:
            self.x_dot = nn.Sequential(nn.Linear(int(3*3*hidden_dim), hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        else:
            self.x_dot = nn.Sequential(nn.Linear(int(3*4*hidden_dim), hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t0: torch.Tensor, xt: torch.Tensor, zt: torch.Tensor, vt: torch.Tensor, it: torch.Tensor, all_initial: torch.Tensor):
        xt_all = torch.cat((xt, zt, vt, it), dim=-1)
        return self.x_dot.forward(torch.cat((all_initial, xt_all-all_initial, xt_all), dim=-1))


class AE_Func(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, v_dim: int, i_dim: int, hidden_dim: int):
        super(AE_Func, self).__init__()
        if z_dim == 0:
            self.i_calculator = nn.Sequential(nn.Linear(int((3+2)*hidden_dim), hidden_dim), nn.ELU(),
                                              nn.Linear(hidden_dim, hidden_dim))
        else:
            self.i_calculator = nn.Sequential(nn.Linear(int((4+3)*hidden_dim), hidden_dim), nn.ELU(),
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
        self.z_encoder = nn.Sequential(nn.Linear(z_dim+v_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim)) if z_dim != 0 else None
        self.v_encoder = nn.Sequential(nn.Linear(v_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.i_encoder = nn.Sequential(nn.Linear(i_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, hidden_dim))
        self.i_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                       nn.Linear(hidden_dim, i_dim))
        self.de_func = DE_Func(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, i_dim=i_dim, hidden_dim=hidden_dim)
        self.ae_func = AE_Func(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, i_dim=i_dim, hidden_dim=hidden_dim)
        self.solver = Euler()
        self.event = DAE_Event()
        
    def forward(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor, i: torch.Tensor, event_t: torch.Tensor, z_jump: torch.Tensor, v_jump: torch.Tensor):
        Xh = self.x_encoder(x).permute(1,0,2)
        Zh = z.permute(1,0,2) if self.z_encoder is None else self.z_encoder(z).permute(1,0,2)
        Vh = self.v_encoder(v).permute(1,0,2)
        Ih = self.i_encoder(i).permute(1,0,2)
        all_initial = torch.cat((Xh[0], Zh[0], Vh[0], Ih[0]), dim=-1)
        Zh_jump = z_jump if self.z_encoder is None else self.z_encoder(z_jump)
        Vh_jump = self.v_encoder(v_jump)
        self.event.set_event(t=event_t, z=Zh_jump, v=Vh_jump)
        # batch, time, variable -> time, batch, variable
        Xh_solution, Ih_solution = self.solver.integrate_DAE(x_func=self.de_func,
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
        return self.x_decoder(Xh_solution).permute(1, 0, 2), self.i_decoder(Ih_solution).permute(1, 0, 2)
    
    def save_model(self, path: pathlib.Path):
        if not path.exists(): path.mkdir()
        with open(str(path / 'dim.txt'), 'w') as f:
            f.write(str(self.hidden_dim))
        sm = torch.jit.script(self.x_encoder)
        sm.save(str(path / 'x_encoder.pt'))
        sm = torch.jit.script(self.x_decoder)
        sm.save(str(path / 'x_decoder.pt'))
        sm = torch.jit.script(self.z_encoder)
        sm.save(str(path / 'z_encoder.pt'))
        sm = torch.jit.script(self.v_encoder)
        sm.save(str(path / 'v_encoder.pt'))
        sm = torch.jit.script(self.i_encoder)
        sm.save(str(path / 'i_encoder.pt'))
        sm = torch.jit.script(self.i_decoder)
        sm.save(str(path / 'i_decoder.pt'))
        sm = torch.jit.script(self.de_func)
        sm.save(str(path / 'de_func.pt'))
        sm = torch.jit.script(self.ae_func)
        sm.save(str(path / 'ae_func.pt'))


def evalute_model(model: DAE_Model, eval_dataset: DAE_Curves_Sample, eval_dataloader: DataLoader, device, logger: Logger, desc='', pic_path : pathlib.Path=None):
    # change to eval
    model.eval()
    # evaluation
    x_loss = 0.0
    x_loss_per_sample = None
    i_loss = 0.0
    i_loss_per_sample = None
    mask_sum = 0.0
    for i_batch, data_batch in enumerate(tqdm(eval_dataloader, desc=desc, leave=True, ncols=ncols)):
        # transfer to device
        sample_batched = [d.to(device) for d in data_batch]
        # parse
        t, x, z, v, i, event_t, z_jump, v_jump, mask = sample_batched
        
        # forward
        x_pred, i_pred = model.forward(t=t, x=x, z=z, v=v, i=i, event_t=event_t, z_jump=z_jump, v_jump=v_jump)
        
        # cal loss
        # x loss
        losses = Loss_func(x_pred * mask, x * mask, reduction='none')
        tmp_e = torch.sum(losses, axis=1).cpu().detach().numpy()
        x_loss += np.sum(tmp_e)
        x_loss_per_sample = tmp_e if x_loss_per_sample is None else np.cat([x_loss_per_sample, tmp_e], dim=-1)
        # i loss
        losses = Loss_func(i_pred * mask, i * mask, reduction='none')
        tmp_e = torch.sum(losses, axis=1).cpu().detach().numpy()
        i_loss += np.sum(tmp_e)
        i_loss_per_sample = tmp_e if i_loss_per_sample is None else np.cat([i_loss_per_sample, tmp_e], dim=-1)
        # mask sum
        mask_sum += torch.sum(mask).cpu().detach().item()
    # print to logger
    x_loss /= mask_sum
    i_loss /= mask_sum
    logger.testing_log(desc + f': x_loss: {x_loss:14.10f}, i_loss: {i_loss:14.10f}.')

    if pic_path is not None:
        if not pic_path.exists(): pic_path.mkdir()

        data_name = eval_dataset.data_name
        size = 10
        mpl.rcParams['xtick.labelsize'] = size
        mpl.rcParams['ytick.labelsize'] = size
        logger.testing_log('Picture Drawing')
        logger.testing_log('======================================================================================')

        x_pred = x_pred.cpu().detach().numpy()
        i_pred = i_pred.cpu().detach().numpy()
        t = eval_dataset.t.cpu().numpy()
        x = eval_dataset.x.cpu().numpy()
        i = eval_dataset.i.cpu().numpy()
        
        drawn_pic = 0
        for sample_no, tt, xx, ii, pred_xx, pred_ii in zip(range(len(t)), t, x, i, x_pred, i_pred):
            if tt[-1] == -1: continue
            if tt[-1] != -1: fin_step = tt.shape[0]
            else: fin_step = np.where(tt == -1)[0][0]
            # if sample_no != 723: continue
            cur_path = pic_path / f'Sample_{sample_no}'
            if not cur_path.exists(): cur_path.mkdir()
            for d_name, true_value, pred_value in zip(data_name,
                                                      np.concatenate((xx, ii), axis=1).transpose()[:, :fin_step],
                                                      np.concatenate((pred_xx, pred_ii), axis=1).transpose()[:, :fin_step]):
                plt.grid()
                plt.title(f'{d_name[0]}_Epoch_{desc}', fontsize=size)
                plt.xlabel('Time (s)', fontsize=size)
                plt.ylabel(f'{d_name[0]} ({d_name[1]})', fontsize=size)
                plt.plot(tt[:fin_step], true_value, 'b-', label="True value", linewidth=line_width, markersize=mark_size)
                plt.plot(tt[:fin_step], pred_value, 'r--', label="Predicted value", linewidth=line_width, markersize=mark_size)
                plt.legend(fontsize=size)
                plt.savefig(cur_path / f'{d_name[0]}_error_{desc}.jpg', dpi=300, format='jpg')
                plt.clf()
                logger.testing_log(f'{d_name[0]} error: total({sum(np.abs(true_value-pred_value)):12.8f} {d_name[1]}), ' + 
                                    f'average({sum(np.abs(true_value-pred_value))/tt.shape[0]:12.8f} {d_name[1]}), ' + 
                                    f'max_error({max(np.abs(true_value-pred_value)):12.8f} {d_name[1]}), ' + 
                                    f'min_error({min(np.abs(true_value-pred_value)):12.8f} {d_name[1]})')
            logger.testing_log('--------------------------------------------------------------------------------------')
            drawn_pic += 1
            if drawn_pic >= pic_num: break
        plt.close()
    
    # return errors
    return np.array([x_loss, i_loss, x_loss_per_sample, i_loss_per_sample], dtype=object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='Choose device to be used, "gpu" or "cpu". Default value is "cpu".',
                        default='cpu')
    parser.add_argument('--id', type=int,
                        help='If using gpu, choose which gpu to be used. Default value is 0.',
                        default=0)
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
    parser.add_argument('--train_data', type=str,
                        help='Training data File Path',
                        required=False, default='./results/samples_neural_gen_2_training.npz')
    parser.add_argument('--test_data', type=str,
                        help='Testing data File Path',
                        required=False, default='./results/samples_neural_gen_2_testing.npz')
    parser.add_argument('--model', type=str,
                        help='<>\tModel dump/load path, directory can be automatically created, but file must exists.',
                        required=False, default='./models')
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
        training_dataset = DAE_Curves_Sample(data_path=args.train_data, device=device, num_sample=training_sample_num)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        testing_dataset = DAE_Curves_Sample(data_path=args.test_data, device=device)
        testing_dataloader = DataLoader(testing_dataset, batch_size=testing_sample_num, shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=training_dataset.x.shape[-1], z_dim=training_dataset.z.shape[-1],
                          v_dim=training_dataset.v.shape[-1], i_dim=training_dataset.i.shape[-1],
                          hidden_dim=hidden_dim).to(device)
        opt_Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt_Adam, step_size=sch_step, gamma=sch_gamma)
        
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
        logger.training_log(f'batch_size_{batch_size}, learning_rate_{learning_rate}, hidden_dim_{hidden_dim}, lamda_x_{lamda_x_loss}，sch_step_{sch_step}, sch_gamma_{sch_gamma}')
        logger.training_log(f'Use training data: {args.train_data}')
        logger.training_log(f'Use testing data: {args.test_data}')

        # initial model testing
        logger.testing_log('======================================================================================')
        logger.testing_log(f'Initial evaluate on training set.')
        if args.drawing: pic_path = pathlib.Path(model_path / 'pics')
        else: pic_path = None
        eval_error_list.append(evalute_model(model, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Testing_Epoch_0', pic_path=pic_path))
        logger.testing_log('======================================================================================')

        # start training
        logger.training_log('Start training 2nd-order Neural Generator Model')
        logger.training_log('======================================================================================')
        
        for epoch in tqdm(range(1, num_epoch+1), desc='Epoch', ncols=ncols):
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
                x_pred, i_pred = model.forward(t=t, x=x, z=z, v=v, i=i, event_t=event_t, z_jump=z_jump, v_jump=v_jump)
                
                # cal loss
                x_loss = (  torch.sum(Loss_func(x_pred, x, reduction='none') * mask)
                        #   + torch.sum(Loss_func(x_pred[:,:,1:2], x[:,:,1:2], reduction='none') * mask) * 100
                        #   + torch.sum(Loss_func(x_pred[:,:,2:6], x[:,:,2:6], reduction='none') * mask) * 10
                          ) / torch.sum(mask)
                i_loss = torch.sum(Loss_func(i_pred, i, reduction='none') * mask) / torch.sum(mask)
                loss = x_loss + i_loss

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
            eval_error_list.append(evalute_model(model, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Testing_Epoch_{epoch}', pic_path=pic_path))
            logger.testing_log('======================================================================================')

            # save results
            np.savez(str(model_path / 'train_and_eval.npz'), train=train_error_list, eval=eval_error_list, dtype=object)
        # fin
    elif args.testing:
        assert args.model is not None and args.test_data is not None, 'Model or testing set missing! Pleses check.'
        
        # load data
        testing_dataset = DAE_Curves_Sample(args.test_data, device)
        testing_dataloader = DataLoader(testing_dataset, batch_size=testing_sample_num, shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=testing_dataset.x.shape[-1], z_dim=testing_dataset.z.shape[-1],
                          v_dim=testing_dataset.v.shape[-1], i_dim=testing_dataset.i.shape[-1],
                          hidden_dim=hidden_dim).to(device)
        
        # model path
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist！'
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
        evalute_model(model=model, eval_dataset=testing_dataset, eval_dataloader=testing_dataloader, device=device, logger=logger, desc=f'Model {model_path.name} Evaluation', pic_path=pic_path)
        logger.testing_log('======================================================================================')
    elif args.saving:
        assert args.model is not None and args.test_data is not None, 'Model or testing set missing! Pleses check.'
        
        # load data
        testing_dataset = DAE_Curves_Sample(args.test_data, device)
        testing_dataloader = DataLoader(testing_dataset, batch_size=testing_sample_num, shuffle=False)
        
        # build model
        model = DAE_Model(x_dim=testing_dataset.x.shape[-1], z_dim=testing_dataset.z.shape[-1],
                          v_dim=testing_dataset.v.shape[-1], i_dim=testing_dataset.i.shape[-1],
                          hidden_dim=hidden_dim).to(device)
        
        # model path
        model_path = pathlib.Path(args.model)
        assert model_path.is_dir() is False and model_path.exists(), f'{model_path} is not a file or does not exist！'
        model.load_state_dict(torch.load(args.model, map_location=device))
        if args.drawing: pic_path = model_path.parent / 'pics'
        else: pic_path = None

        # save model
        model.save_model(model_path.parent / 'saved model')
        print(f'Model {model_path} saved.')       
    else:
        raise Exception('Unknown task. Set "--training" or "--testing".')
