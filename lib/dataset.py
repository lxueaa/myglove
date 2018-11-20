import os
import nltk
from nltk.tokenize import word_tokenize

import numpy as np
import math
import torch
import gc

# def f(x, x_max, alpha):
# 	fx = torch.where(x<x_max, (x/x_max)**alpha, torch.ones_like(x))
# 	return fx

def f(x, x_max, alpha):
	if x < x_max:
		return (x/x_max)**alpha
	else:
		return 1.0

def import_data(dataset, input_folder, output_folder, window_size, freq, x_max, alpha):
	"""
	Read data from dataset, and save to file.
	:param dataset: the pathto/dataset, string
	:param window_size: size of context
	:return: tokens, vocabs, n_token, n_vocab, dic, \
			co_matrix, nonzeros, n_nonzero
	"""

	root_path = os.getcwd()

	output_path = os.path.join(root_path, output_folder, dataset)

	isTokenExist = os.path.isfile(os.path.join(output_path, 'tokens.txt'))
	isVocabExist = os.path.isfile(os.path.join(output_path, 'vocabs.txt'))
	isDictExist = os.path.isfile(os.path.join(output_path, 'dictionary.txt'))
	isMatrixExist = os.path.isfile(os.path.join(output_path, 'fx_logx_matrix.txt'))
	# isMatrixExist = os.path.isfile(os.path.join(output_path, 'co_matrix_sparse.txt'))

	# create vocabulray and word lists 
	if isTokenExist and isVocabExist and isDictExist and isMatrixExist:  
		print('Already preprocessed, loading from files...')
		n_vocab, dic, fx_logx_matrix = \
		load_data(output_path)
	else:
		n_vocab, dic, fx_logx_matrix = \
		read_data(dataset, root_path, input_folder, output_folder, window_size, freq, x_max, alpha)
	
	print('#vocabs: ', n_vocab)
	print('#nonzeros: ', len(fx_logx_matrix))

	return n_vocab, dic, fx_logx_matrix


def read_data(dataset, root_path, input_folder, output_folder, window_size, freq, x_max, alpha):
	"""
	Read data from dataset, and save to file.
	:param dataset: the pathto/dataset, string
	:param window_size: size of context
	:return: tokens, vocabs, n_token, n_vocab, dic, \
			co_matrix, nonzeros, n_nonzero
	"""

	print('\nReading data from dataset...')

	input_path = os.path.join(root_path, input_folder, dataset)

	print(input_path)

	text = ''
	tokens = []
	vocabs = {}

	# create vocabulray and word lists 
	if os.path.isfile(input_path):  
		text_file = open(input_path,'r')
		text += text_file.read().lower()
		text_file.close()
		   
		tokens = word_tokenize(text)
		for token in tokens:
			if token not in vocabs:
				vocabs[token] = 1
			else:
				vocabs[token] += 1

	elif os.path.isdir(input_path):  
		filelist = os.listdir(input_path)
		for item in filelist:
			filename = os.path.join(input_path, item)
			print(" - file: ", item)

			if os.path.isfile(filename): 
				text_file = open(filename, 'r')
				text = text_file.read().lower()
				token_list = word_tokenize(text)
				
				for token in token_list:
					tokens.append(token)
					if token not in vocabs:
						vocabs[token] = 1
					else:
						vocabs[token] += 1

				text_file.close()

	# create vocabulray and word lists 
	tokens = [word for word in tokens if vocabs[word]>freq]
	vocabs = sorted(word[0] for word in vocabs.items() if vocabs[word[0]]>freq)
	

	n_token = len(tokens)
	n_vocab = len(vocabs)
	print('#tokens: ', n_token)
	print('#vocabs: ', n_vocab)

	print('\tcreate indices......')
	dic = {word:vid for vid, word in enumerate(vocabs)}
	
	print('\tconstruct co-matrix......')

	'''	
	co_matrix_sparse = {}
	co_matrix = torch.zeros((n_vocab, n_vocab))
	for current in range (n_token):
		for context in range (1, window_size+1):
			id_current = dic[tokens[current]]
			left = current - context
			right = current + context
			# left hand side
			if left >=0:
				id_left = dic[tokens[left]]
				index = tuple((id_current, id_left))
				if index in co_matrix_sparse:
					co_matrix_sparse [index] += 1.0/context#{ 1.0/context if weighted_distance else 1.0}
				else:
					co_matrix_sparse [index] = 1.0/context
			#right hand side
			if right < n_token:
				id_right = dic[tokens[right]]
				index = tuple((id_current, id_right))
				if index in co_matrix_sparse:
					co_matrix_sparse [index] += 1.0/context#{ 1.0/context if weighted_distance else 1.0}
				else:
					co_matrix_sparse [index] = 1.0/context
	'''

	fx_logx_matrix = {}
	for current in range (n_token):
		for context in range (1, window_size+1):
			id_current = dic[tokens[current]]
			left = current - context
			right = current + context
			# left hand side
			if left >=0:
				id_left = dic[tokens[left]]
				index = tuple((id_current, id_left))
				if index in fx_logx_matrix:
					fx_logx_matrix [index][0] += 1.0/context
					fx_logx_matrix [index][1] += 1.0/context
				else:
					fx_logx_matrix [index] = [1.0/context, 1.0/context]
			
			#right hand side
			if right < n_token:
				id_right = dic[tokens[right]]
				index = tuple((id_current, id_right))
				if index in fx_logx_matrix:
					fx_logx_matrix [index][0] += 1.0/context
					fx_logx_matrix [index][1] += 1.0/context
				else:
					fx_logx_matrix [index] = [1.0/context, 1.0/context]
	
	# find out non-zero entities in matrix
	# nonzeros = torch.nonzero(co_matrix)
	# nonzeros = list(co_matrix_sparse)
	
	n_nonzero = len(fx_logx_matrix)
	print('#nonzeros: ', n_nonzero)


	print('\nSaving data to files...')

	output_path = os.path.join(root_path, output_folder, dataset)

	save_list(tokens, output_path, 'tokens.txt')
	del tokens
	gc.collect()

	save_list(vocabs, output_path, 'vocabs.txt')
	del vocabs
	gc.collect()

	save_dict(dic, output_path, 'dictionary.txt')

	print('\tconstruct dense co-matrix......')

	for key, value in fx_logx_matrix.items():
		if value[0] < x_max:
			value[0] = (value[0]/x_max)**alpha
		else:
			value[0] = 1.0
		value[1] = math.log(value[1])
		fx_logx_matrix[key] = value

	save_fx_logx_matrix(fx_logx_matrix, output_path, 'fx_logx_matrix.txt')

	# fx_logx_matrix = {item[0]:[ f(item[1][0], x_max, alpha), np.log(item[1][1]+1)] for item in fx_logx_matrix.items() }

	# x = torch.tensor([co_matrix_sparse[item] for item in nonzeros])

	# fx = f(x, x_max, alpha)
	# logx = np.log(x)

	# fx_logx_matrix = {nonzeros[i]:[fx[i],logx[i]] for i in range(n_nonzero)}

	# save_matrix(co_matrix_sparse, output_path, 'co_matrix_sparse.txt')
	# save_matrix(nonzeros, root_path, 'data/', 'nonzeros.txt')

	return n_vocab, dic, fx_logx_matrix


def save_list(data, path, name):
	"""
	Save data to file:

	:param data: data want to save
	:param name: name of file
	:return: returns nothing
	"""

	print("\tsaving ", name, " to files..." )
	output_path = path
	is_exist = os.path.exists(output_path)
	
	if not is_exist:
		os.makedirs(output_path)

	filename = output_path + "/" + name

	output_file = open(filename, "w")

	# if want to keep structure and visual friendly, choose this
	for item in data:
		newline = item + '\n'
		output_file.write(newline)

	# if want to be fast and easy, choose this
	# output_file = open(filename, "w")
	# output_file.write(str(data))
	
	output_file.close()


def save_dict(data, path, name):
	"""
	Save data to file:

	:param data: data want to save
	:param name: name of file
	:return: returns nothing
	"""

	print("\tsaving ", name, " to files..." )

	output_path = path
	is_exist = os.path.exists(output_path)
	
	if not is_exist:
		os.makedirs(output_path)

	filename = output_path + "/" + name

	output_file = open(filename, "w")

	# if want to keep structure and visual friendly, choose this
	for item in data:
		newline = ' '.join((item, str(data[item]))) + '\n'
		#newline = str(item) + '\n'
		output_file.write(newline)

	# if want to be fast and easy, choose this
	# output_file = open(filename, "w")
	# output_file.write(str(data))

	output_file.close()


def save_matrix(data, path, name):
	"""
	Save data to file:

	:param data: data want to save
	:param name: name of file
	:return: returns nothing
	"""

	print("\tsaving ", name, " to files..." )

	output_path = path
	is_exist = os.path.exists(output_path)
	
	if not is_exist:
		os.makedirs(output_path)

	filename = output_path + "/" + name

	output_file = open(filename, "w")

	# if want to keep structure and visual friendly, choose this
	for item in data:
		newline = ' '.join(( str(list(item)[0]), str(list(item)[1]), str(data[item]))) + '\n'
		output_file.write(newline)

	# if want to be fast and easy, choose this
	# output_file = open(filename, "w")
	# output_file.write(str(data))

	output_file.close()

def save_fx_logx_matrix(data, path, name):
	print("\tsaving ", name, " to files..." )

	output_path = path
	is_exist = os.path.exists(output_path)
	
	if not is_exist:
		os.makedirs(output_path)

	filename = output_path + "/" + name

	output_file = open(filename, "w")

	# if want to keep structure and visual friendly, choose this
	for item in data:
		newline = ' '.join( (str(list(item)[0]), str(list(item)[1]), \
			str('%.5f'% data[item][0]), str('%.5f'% data[item][1]))) + '\n'
		output_file.write(newline)

def load_data(path):
	"""
	Load data from file:

	:param data: data want to save
	:param name: name of file
	:return: data
	"""
	print("\nLoading data...")

	# load_tokens(data, path)
	
	names = {'vocabs', 'dictionary', 'fx_logx_matrix'}#'co_matrix_sparse'}

	for name in names:
		filename = path + "/" + name + '.txt'
		# if name == 'vocabs':
		# 	vocabs = load_vocabs(filename)
		if name == 'dictionary':
			dic = load_dict(filename)
		if name == 'fx_logx_matrix':
			fx_logx_matrix = load_fx_logx_matrix(filename) 

		# if name == 'co_matrix_sparse':
		# 	co_matrix_sparse = load_matrix(filename)

	"""
	for name in names:
		file = open(path + "/" + name + '.txt', "r")
		content = file.read()

		if name == 'tokens':
			print('\tloading tokens...')
			tokens = eval(content)

		elif name == 'vocabs':
			print('\tloading vocabs...')
			vocabs = eval(content)

		elif name == 'dictionary':
			print('\tloading dictionary...')
			dic = eval(content)

		elif name == 'co_matrix_sparse':
			print('\tloading matrix...')
			co_matrix_sparse = eval(content)

		file.close()
	"""
	print('Finish loading...Collecting...')

	# nonzeros = list(co_matrix_sparse)

	# n_token = len(tokens)
	# nonzeros = list(fx_logx_matrix)
	n_vocab = len(dic)
	# n_nonzero = len(nonzeros)

	# print(nonzeros[0])

	return n_vocab, dic, fx_logx_matrix

def load_tokens(path):

	print('\tloading tokens......')
	
	tokens = []
	f = open(path, 'r')
	for line in f.readlines():
		tokens.append(line)

	f.close()
	return tokens

def load_vocabs(path):

	print('\tloading vocabs......')

	vocabs = []
	f = open(path, 'r')
	for line in f.readlines():
		vocabs.append(line)

	f.close()
	return vocabs

def load_dict(path):

	print('\tloading dictionary......')

	dic = {}
	f = open(path, 'r')
	for line in f.readlines():
		seq = line.split()
		dic[seq[0]] = int(seq[1])

	f.close()
	return dic

def load_matrix(path):

	print('\tloading matrix......')

	matrix = {}
	f = open(path, 'r')
	for line in f.readlines():
		seq = line.split()
		index = tuple((int(seq[0]), int(seq[1])))
		matrix[index] = float(seq[2])

	f.close()
	return matrix

def load_fx_logx_matrix(path):

	print('\tloading matrix......')

	matrix = {}
	f = open(path, 'r')
	for line in f.readlines():
		seq = line.split()
		index = tuple((int(seq[0]), int(seq[1])))
		matrix[index] = [float(seq[2]), float(seq[3])]

	f.close()
	return matrix

def save_embeddings(input_vectors, output_vectors, dataset, output_folder):
	"""
	"""
	print("\nSave embeddings to files...")
	
	# save leart embeddings to file 'embed'/
	root_path = os.getcwd()
	save_path = os.path.join(root_path, output_folder, dataset)

	n_vocab = len(input_vectors)

	fu = open(save_path + '/input_vectors.txt', "w")
	for i in range(n_vocab):
		line = input_vectors[i]
		newline = ' '.join(str('%.5f' % j) for j in line) + '\n'
		fu.write(newline)
	fu.close()

	fv = open(save_path + '/output_vectors.txt', "w")
	for i in range(n_vocab):
		line = output_vectors[i]
		newline = ' '.join(str('%.5f' % j) for j in line) + '\n'
		fv.write(newline)
	fv.close()

	# np.savetxt(save_path + '/input_vectors.txt', input_vectors, fmt='%0.5f')
	# np.savetxt(save_path + '/output_vectors.txt', output_vectors, fmt='%0.5f')


def load_embeddings(fu_name, fv_name):
	"""
	"""
	print("\nLoading embeddings..." )

	input_vectors, output_vectors = [], []

	fu = open(fu_name, 'r')
	for line in fu.readlines():
		line = line.split()
		array = [float(i) for i in line]
		input_vectors.append(array)
	fu.close()

	fv = open(fv_name, 'r')
	for line in fv.readlines():
		line = line.split()
		array = [float(i) for i in line]
		output_vectors.append(array)
	fv.close()

	return input_vectors, output_vectors





