import numpy as np
import torch 

class Dataloader(object):
	'''data loader for glove'''
	def __init__(self, matrix):#, n_nonzero, nonzeros):
		super(Dataloader, self).__init__()
		self.matrix = matrix
		# self.u, self.v, self.fx_list, self.logx_list = [], [], [], []
		# for key, value in matrix.items():
		# 	self.u.append(key[0])
		# 	self.v.append(key[1])
		# 	self.
		self.u = np.array([item[0][0] for item in matrix.items()])
		self.v = np.array([item[0][1] for item in matrix.items()])
		# print(matrix[(self.u[0],self.v[0])])
		self.fx_list = np.array([item[1][0] for item in matrix.items()])
		self.logx_list = np.array([item[1][1] for item in matrix.items()])
		self.n_nonzero = len(matrix)

	def get_batch(self, batch_size):
		"""
		Generate batches: return indices of input vectors and outputvectors,
		and word-word co-occurrences corresponding to u-v pairs

		:param batch_size: number of samples in a batch
		:return: uid, vid, x: 
		:rtype: tensor
		"""
		# sample = np.random.choice(np.arange(self.n_nonzero), size = batch_size, replace = False)
		sample = torch.randint(0, self.n_nonzero, (batch_size,), dtype=torch.int)

		uid = self.u[sample]
		vid = self.v[sample]
		fx = self.fx_list[sample]
		logx = self.logx_list[sample]

		# uid, vid, fx, logx = [], [], [], []
		# for chosen in sample:
		# 	tmp = tuple(self.nonzeros[chosen])
		# 	uid.append(tmp[0])
		# 	vid.append(tmp[1])
		# 	fx.append(self.matrix[tmp][0])
		# 	logx.append(self.matrix[tmp][1])
		# 	# print(tmp[0], '\t', tmp[1], '\t', self.co_matrix[tmp])
		
		uid = torch.from_numpy(uid)
		vid = torch.from_numpy(vid)
		fx = torch.from_numpy(fx)
		logx = torch.from_numpy(logx)
		return uid, vid, fx, logx








