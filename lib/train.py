import time
import os

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

import numpy as np
import math
import torch 
import torch.nn as nn
import torch.optim as optim
import lib.dataset as ds

def main(model, device, dataloader, x_max, alpha, lr, batch_size, n_epoch, dataset_name, output_folder):
	
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model, device_ids=[0, 1])	
	
	model.to(device)
	# print('after model to device: ', torch.cuda.memory_allocated())
		
	n_nonzero = dataloader.n_nonzero

	optimizer = optim.SparseAdam(model.parameters(), lr=lr)
	# optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=1e-3)
	losses=[]

	# plt.ion()
	# plt.show()

	# batch_size = n_nonzero

	n_batch = math.ceil(n_nonzero / batch_size)
	# n_batch = 100
	n_iter = n_epoch * n_batch

	print('#epoch: ', n_epoch)
	print('#batch: ', n_batch, ' | batch_size: ', batch_size)
	print('#iter: ', n_iter)

	avg_loss = 0.0
	epoch = 1
	batch = 0

	start_time = time.time()
	tic = start_time
	tic_batch = start_time

	# isPklExist = os.path.isfile('./net_params.pkl')
	isPklExist = os.path.isfile('./net.pkl')
	
	if isPklExist:
		# model.load_state_dict(torch.load('./net_params.pkl'))
		model = torch.load('./net.pkl') 

	print("\nTraining start >>>>>>>>>>>>>>>")

	for it in range (n_iter):
		# plt.cla()
		batch += 1
		loss = 0.0

		uid, vid, fx, logx = dataloader.get_batch(batch_size)
				
		uid = uid.to(device)
		vid = vid.to(device)
		# x = x.to(device)

		# fx = self.f(x, x_max, alpha)
		# logx = np.log(x)
			
		fx = fx.to(device)
		logx = logx.to(device)

		optimizer.zero_grad()
		loss = model.forward(uid, vid, logx, fx)

		del uid, vid, logx, fx

		loss.sum().backward()
		optimizer.step()

		avg_loss += loss.sum().item() 

		if (batch) % 10000 == 0:
			toc_batch = time.time() - tic_batch
			print("\tepoch %3d/%d, batch %5d/%d, loss = %5f " \
				% (epoch, n_epoch, batch, n_batch, loss.sum()),'time span',\
				time.strftime("%H:%M:%S", time.gmtime(toc_batch)))
			tic_batch = time.time()

		if (it+1) % n_batch == 0:
			avg_loss /= n_batch
			losses.append(avg_loss)

			toc = time.time() - tic
			print("Epoch %3d/%d, average loss = %5f, "\
				 % (epoch, n_epoch, avg_loss), 'time span',\
				 time.strftime("%H:%M:%S", time.gmtime(toc)))

			avg_loss == 0.0
			epoch += 1
			batch = 0
			
			
			input_vectors = list(model.input.weight.data.cpu().numpy())
			output_vectors = list(model.output.weight.data.cpu().numpy())

			torch.save(model, './net.pkl')
			'''
			torch.save(model.state_dict(), './net_params.pkl')
			'''

			ds.save_embeddings(input_vectors, output_vectors, dataset_name, output_folder)

			tic = time.time()

			if epoch > 10 and losses[epoch-3] - losses[epoch-2] <= 0.00001:
				lr = lr / 4.0

			if epoch % 1 == 0:
				l_r = lr/(epoch)
				optimizer = optim.SparseAdam(model.parameters(), lr = l_r)

			
			# plt.plot(losses, color='pink', marker='o', linestyle='dashed', linewidth=2, markersize=12)
			# plt.pause(0.1)

		# plt.ioff()
		# plt.show()
		# plt.plot(losses, color='pink', marker='o', linestyle='dashed', linewidth=2, markersize=12)
		# plt.savefig("img/loss.png")

	elapsed_time = time.time() - start_time
	print('>>>>>>>>>>>>>>> Time cost in training: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

	input_vectors = list(model.input.weight.data.cpu().numpy())
	output_vectors = list(model.output.weight.data.cpu().numpy())

	return losses, input_vectors, output_vectors
