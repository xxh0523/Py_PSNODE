from sys import maxsize
import numpy as np
from numpy.core.fromnumeric import size
import torch
import pathlib
from tqdm import tqdm


class Logger:
    def __init__(self, logfile_path: pathlib.Path, train_log_name=None, test_log_name=None):
        self.training_logfile = None if train_log_name is None else open(logfile_path / train_log_name, 'w')
        self.testing_logfile = None if test_log_name is None else open(logfile_path / test_log_name, 'w')
    
    def __del__(self):
        if self.training_logfile is not None: self.training_logfile.close()
        if self.testing_logfile is not None: self.testing_logfile.close()
    
    def training_log(self, *strs):
        string = ' '.join(strs)
        self.training_logfile.write(string + '\n')
        tqdm.write(string)
    
    def testing_log(self, *strs):
        string = ' '.join(strs)
        self.testing_logfile.write(string + '\n')
        tqdm.write(string) 


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.p = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		# ind[0] = np.argmax(self.reward)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def importance_sampling(self, batch_size):
		between_01 = np.where(self.reward > 0)[0]
		if len(between_01) > 0:
			po = 0.5 / len(between_01)
			pd = 0.5 / (self.size - len(between_01))
			for i in range(self.size):
				self.p[i] = po if i in between_01 else pd
		else:
			self.p[:self.size].fill(1. / self.size)
		index = np.arange(self.size)
		ind = np.random.choice(index, size=batch_size, p=self.p[:self.size,:].reshape(-1))
		ind[0] = np.argmax(self.reward.reshape(-1))

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)	
