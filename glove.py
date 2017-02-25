import sys, time, random

import numpy as np

import theano
import theano.tensor as T
from theano import config
from theano.ifelse import ifelse

import cPickle as pickle

from collections import OrderedDict

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def unzip(zipped):
	new_params = OrderedDict()
	for k, v in zipped.iteritems():
		new_params[k] = v.get_value()
	return new_params

def init_params(options):
	params = OrderedDict()

	inputSize = options['inputSize']
	dimensionSize = options['dimensionSize']

	rng = np.random.RandomState(1234)
	params['w'] = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(inputSize, dimensionSize)), dtype=theano.config.floatX)
	rng = np.random.RandomState(12345)
	params['w_tilde'] = np.asarray(rng.uniform(low=-0.1, high=0.1, size=(inputSize, dimensionSize)), dtype=theano.config.floatX)

	params['b'] = np.zeros(inputSize).astype(theano.config.floatX)
	params['b_tilde'] = np.zeros(inputSize).astype(theano.config.floatX)

	return params

def init_tparams(params):
	tparams = OrderedDict()
	for k, v in params.iteritems():
		tparams[k] = theano.shared(v, name=k)
	return tparams

def build_model(tparams, options):
	weightVector = T.vector('weightVector', dtype=theano.config.floatX)
	iVector = T.vector('iVector', dtype='int32')
	jVector = T.vector('jVector', dtype='int32')
	cost = weightVector * (((tparams['w'][iVector] * tparams['w_tilde'][jVector]).sum(axis=1) + tparams['b'][iVector] + tparams['b_tilde'][jVector] - T.log(weightVector)) ** 2)

	return weightVector, iVector, jVector, cost.sum()

def adadelta(tparams, grads, weightVector, iVector, jVector, cost):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([weightVector, iVector, jVector], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update

def weightFunction(x):
	if x < 100.0:
		return (x / 100.0) ** 0.75
	else:
		return 1

def load_data(infile):
	cooccurMap = pickle.load(open(infile, 'rb'))
	I = []
	J = []
	Weight = []
	for key, value in cooccurMap.iteritems():
		I.append(key[0])
		J.append(key[1])
		Weight.append(weightFunction(value))
	shared_I = theano.shared(np.asarray(I, dtype='int32'), borrow=True)
	shared_J = theano.shared(np.asarray(J, dtype='int32'), borrow=True)
	shared_Weight = theano.shared(np.asarray(Weight, dtype=theano.config.floatX), borrow=True)
	return shared_I, shared_J, shared_Weight

def print2file(buf, outFile):
	outfd = open(outFile, 'a')
	outfd.write(buf + '\n')
	outfd.close()

def train_glove(infile, inputSize=20000, batchSize=100, dimensionSize=100, maxEpochs=1000, outfile='result', x_max=100, alpha=0.75):
	options = locals().copy()
	print 'initializing parameters'
	params = init_params(options)
	tparams = init_tparams(params)

	print 'loading data'
	I, J, Weight = load_data(infile)
	n_batches = int(np.ceil(float(I.get_value(borrow=True).shape[0]) / float(batchSize)))

	print 'building models'
	weightVector, iVector, jVector, cost = build_model(tparams, options)
	grads = T.grad(cost, wrt=tparams.values())
	f_grad_shared, f_update = adadelta(tparams, grads, weightVector, iVector, jVector, cost)

	logFile = outfile + '.log'
	print 'training start'
	for epoch in xrange(maxEpochs):
		costVector = []
		iteration = 0
		for batchIndex in random.sample(range(n_batches), n_batches):
			cost = f_grad_shared(Weight.get_value(borrow=True, return_internal_type=True)[batchIndex*batchSize:(batchIndex+1)*batchSize],
								I.get_value(borrow=True, return_internal_type=True)[batchIndex*batchSize: (batchIndex+1)*batchSize],
								J.get_value(borrow=True, return_internal_type=True)[batchIndex*batchSize: (batchIndex+1)*batchSize])
			f_update()
			costVector.append(cost)

			if (iteration % 1000 == 0):
				buf = 'epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, cost)
				print buf
				print2file(buf, logFile)
			iteration += 1
		trainCost = np.mean(costVector)
		buf = 'epoch:%d, cost:%f' % (epoch, trainCost)
		print buf
		print2file(buf, logFile)
		tempParams = unzip(tparams)
		np.savez_compressed(outfile + '.' + str(epoch), **tempParams)

def get_rootCode(treeFile):
	tree = pickle.load(open(treeFile, 'rb'))
	return tree.values()[0][1]

if __name__=='__main__':
	infile = sys.argv[1]
	treeFile = sys.argv[2]
	outfile = sys.argv[3]

	inputDimSize = get_rootCode(treeFile+'.level2.pk') + 1
	embDimSize = 128
	batchSize = 100
	maxEpochs = 50
	train_glove(infile, inputSize=inputDimSize, batchSize=batchSize, dimensionSize=embDimSize, maxEpochs=maxEpochs, outfile=outfile)
