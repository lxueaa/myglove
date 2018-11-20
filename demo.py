from nltk.tokenize import word_tokenize
import matplotlib
import matplotlib.pyplot as plt
import os
#matplotlib.use('Agg')

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import lib.train as train

import lib.dataset as ds
import lib.evaluation as evaluation
from lib.utils import Dataloader
from lib.model import Glove
# import lib.plot


torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set parameters
embed_dim = 300 # embedding dimension, default 300
window_size = 10# default 10 
x_max = 100.0 # default 100.0 
freq = 10
alpha = 0.75

lr = 0.05 # default 0.05 at beginning
batch_size = 512 # 512
n_epoch = 25 # 50 when d <= 300; otherwise, 100 

weighted_dist = False # if True, +=1.0/distance; else, += 1.0

input_folder = 'dataset'
output_folder = 'data/'
dataset_name = 'text8'

'''whether embedding is learnt'''
load_path = os.path.join(os.getcwd(), output_folder, dataset_name)

fu_name = load_path + '/input_vectors.txt'
fv_name = load_path + '/output_vectors.txt'

isFileExist = os.path.isfile(fu_name) and os.path.isfile(fv_name)

if isFileExist:
	path = os.path.join(os.getcwd(), output_folder, dataset_name)
	dic = ds.load_dict(path+'/dictionary.txt')
	input_vectors, output_vectors = ds.load_embeddings(fu_name, fv_name)

else:
	n_vocab, dic, fx_logx_matrix = \
	ds.import_data(dataset_name, input_folder, output_folder, window_size, freq, x_max, alpha)

	dataloader = Dataloader(fx_logx_matrix)

	glove = Glove(n_vocab=n_vocab, embed_dim=embed_dim)

	# batch_size = len(fx_logx_matrix)

	losses, input_vectors, output_vectors = train.main(model = glove,\
		device=device, dataloader=dataloader, x_max=x_max, alpha=alpha, \
		lr=lr, batch_size=batch_size, n_epoch=n_epoch, \
		dataset_name = dataset_name, output_folder=output_folder)

	# ds.save_embeddings(input_vectors, output_vectors, dataset_name, output_folder)

print('\nEvaluations...')
wordsim_corr = evaluation.wordsim353(input_vectors, dic)
print("\tWordSim353: %.4f" % wordsim_corr)
rw_corr = evaluation.rw(input_vectors, dic)
print("\tRW: %.4f" % rw_corr)
acc = evaluation.analogy(input_vectors, dic)




'''
torch.save(model, './net.pkl')
torch.save(model.state_dict(), './net_params.pkl')
'''
'''
model = torch.load('./net.pkl')
model.load_state_dict(torch.load('./net_params.pkl'))
'''






