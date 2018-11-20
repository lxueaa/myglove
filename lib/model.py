import time

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

class Glove(nn.Module):
	"""word embedding vectors"""
	def __init__(self, n_vocab, embed_dim, sparse=True):
		super(Glove, self).__init__()
		self.n_vocab = n_vocab
		self.embed_dim = embed_dim

		# embedding for current words
		self.input = nn.Embedding(n_vocab, embed_dim, sparse=sparse)
		self.input_bias = nn.Embedding(n_vocab, 1, sparse=sparse)
		
		# embedding for context words
		self.output = nn.Embedding(n_vocab, embed_dim, sparse=sparse)
		self.output_bias = nn.Embedding(n_vocab, 1, sparse=sparse)

		self.init_embeddings()

	def init_embeddings(self):
		'''initialize word embeddings'''
		# initrange = (2.0 / (self.n_vocab + self.embed_dim)) ** 0.5
		initrange = 0.5 / self.embed_dim
		#initrange = (2.0 / (vocab_size + emb_dim))**0.5
		self.input.weight.data.uniform_(-initrange,initrange)
		self.output.weight.data.uniform_(-initrange,initrange)
		self.input_bias.weight.data.zero_()
		self.output_bias.weight.data.zero_()

	def forward(self, uid, vid, logx, fx):
		u, v = self.input(uid), self.output(vid)
		bu, bv = self.input_bias(uid), self.output_bias(vid)
		predict = torch.sum(torch.mul(u, v), dim=1) + bu + bv #.view(bu.size())
		#predict  = torch.mm(u, v.t()) + bu + bv.t()
		loss = torch.mean(0.5 * fx * ((predict - logx) ** 2))
		return loss

	# def f(self, x, x_max, alpha):
	# 	fx = torch.where(x<x_max, (x/x_max)**alpha, torch.ones_like(x))
	# 	return fx

	def train(self, device, dataloader, x_max, alpha, lr, batch_size, n_batch, n_iter):
		self.to(device)
		optimizer = optim.SparseAdam(self.parameters(), lr=lr)
		losses=[]

		# plt.ion()
		# plt.show()

		start_time = time.time()

		print("\nTraining start >>>>>>>>>>>>>>>")

		for it in range (n_iter):
			# plt.cla()
			loss = 0.0
			avg_loss = 0.0
			for batch in range(n_batch):
				uid, vid, fx, logx = dataloader.get_batch(batch_size)
				
				uid = uid.to(device)
				vid = vid.to(device)
				# x = x.to(device)

				# fx = self.f(x, x_max, alpha)
				# logx = np.log(x)
			
				fx = fx.to(device)
				logx = logx.to(device)

				optimizer.zero_grad()
				loss = self.forward(uid, vid, logx, fx)

				loss.backward(retain_graph=True)
				optimizer.step()

				avg_loss += loss / n_batch

				# if batch % 5 == 0:
				# 	print('\tbatch ', batch, ', avg_loss = ', avg_loss)
		
			print("\tIter %d Loss: %.5f" % (it, avg_loss.data))

			losses.append(loss.data)

			if it > 25:
				if losses[it] > losses[it-1]:
					lr = lr/2.0
					optimizer = optim.SparseAdam(self.parameters(), lr=lr) 
			
			# plt.plot(losses, color='pink', marker='o', linestyle='dashed', linewidth=2, markersize=12)
			# plt.pause(0.1)

		# plt.ioff()
		# plt.show()
		# plt.plot(losses, color='pink', marker='o', linestyle='dashed', linewidth=2, markersize=12)
		# plt.savefig("img/loss.png")

		elapsed_time = time.time() - start_time
		print('>>>>>>>>>>>>>>> Time cost in training: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

		input_vectors = list(self.input.weight.data.cpu().numpy())
		output_vectors = list(self.output.weight.data.cpu().numpy())

		return losses, input_vectors, output_vectors
