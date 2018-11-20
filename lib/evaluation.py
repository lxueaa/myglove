import numpy as np
from numpy import linalg as LA
import torch
import os

import matplotlib.pyplot as plt
import matplotlib
import time

from scipy.stats import spearmanr

def wordsim353(embedding, dic):

	root_path = os.getcwd()
	test_folder = 'testset'
	path = os.path.join(root_path, test_folder, 'wordsim353')
	
	# wordsim, cossim = {}, {}
	rank1,rank2 = [],[]

	f_name = path + '/combined.tab'
	f = open(f_name, 'r')
	cnt = -1
	for line in f.readlines():
		if cnt == -1:
			cnt = cnt + 1
			continue

		seq = line.split()

		if seq[0].lower() not in dic or seq[1].lower() not in dic:
			# print("not exist")
			continue

		uid = dic[seq[0].lower()]
		vid = dic[seq[1].lower()]

		u = embedding[uid]
		v = embedding[vid]

		cos = abs(np.dot(u,v)/(LA.norm(u) * LA.norm(v)))
		rank1.append(float(seq[2]))
		rank2.append(cos)

		# print(uid,',', vid, seq[0].lower(), seq[1].lower(), float(seq[2]), cos)
	print('count: ', len(rank1))
	corr, p = spearmanr(rank1, rank2)
	return corr


def rw(embedding, dic):

	root_path = os.getcwd()
	test_folder = 'testset'
	path = os.path.join(root_path, test_folder, 'rw')
	
	# wordsim, cossim = {}, {}
	rank1,rank2 = [],[]

	f_name = path + '/rw.txt'
	f = open(f_name, 'r')

	for line in f.readlines():

		seq = line.split()

		if seq[0].lower() not in dic or seq[1].lower() not in dic:
			continue

		uid = dic[seq[0].lower()]
		vid = dic[seq[1].lower()]

		u = embedding[uid]
		v = embedding[vid]

		cos = abs(np.dot(u,v)/(LA.norm(u) * LA.norm(v)))
		rank1.append(float(seq[2]))
		rank2.append(cos)
		# print(seq[0].lower(), seq[1].lower(), float(seq[2]), cos)

	print('count: ', len(rank1))
	corr, p_value = spearmanr(rank1, rank2)
	return corr

def analogy(embedding, dic):
	
	root_path = os.getcwd()
	test_folder = 'testset'
	path = os.path.join(root_path, test_folder, 'analogy')
	
	n_vocab = len(dic)
	i2w = {item[1]:item[0] for item in dic.items()}

	base = LA.norm(embedding, ord=2, axis=1)
	embedding = embedding/base.reshape(n_vocab,1)

	f_name = path + '/w2v_analogy.txt'
	f = open(f_name, 'r')

	cnt = 0.0
	correct = 0.0

	for line in f.readlines():

		seq = line.split()
		
		if seq[0] == ':':
			continue

		if seq[0].lower() not in dic or seq[1].lower() not in dic or seq[2].lower() not in dic or seq[3].lower() not in dic:
			continue

		cnt = cnt + 1

		aid = dic[seq[0].lower()]
		bid = dic[seq[1].lower()]
		cid = dic[seq[2].lower()]
		did = dic[seq[3].lower()]

		a = embedding[aid]
		b = embedding[bid]
		c = embedding[cid]
		d = np.subtract(np.add(b, c), a)

		d = d/LA.norm(d)
		cos = np.dot(embedding, d)
		did_pre = np.argmax(cos)

		d_word = i2w[did_pre]

		if did_pre == did:
			correct = correct + 1
			print(seq[0], seq[1], seq[2], seq[3], "\tpredict: ", d_word)

		# if cnt % 500 == 0:
		# 	print(seq[0], seq[1], seq[2], seq[3], "\tpredict: ", d_word)
			# print('sab = ', sab, 'scd = ', scd)
			# print(cos_predict, cos_true, cos_d)
			# print(embedding[did])
			# print(embedding[did_predict])
			# print(d)
	
	print('correct: ', correct, '/', cnt)

	acc = 0.0
	if not cnt == 0:
		acc = correct/cnt

	print('\tWord Analogy, acc = %.4f' % acc)

	return acc

