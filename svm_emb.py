import sys, random
import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

theano.config.cxx = ""

SEED = 1234

def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.iteritems():
        new_params[key] = value.get_value()
    return new_params

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

def init_params(options):
    params = OrderedDict()

    inputDimSize = options['inputDimSize']
    hiddenDimSize = options['hiddenDimSize']

    params['W_emb'] = np.array(pickle.load(open(options['embFile'], 'rb'))).astype(config.floatX)
    params['W_rnn'] = np.random.uniform(-0.01, 0.01, (options['embDimSize'], options['hiddenDimSize']))
    params['U_rnn'] = np.random.uniform(-0.01, 0.01, (options['hiddenDimSize'], options['hiddenDimSize']))
    params['b_rnn'] = np.zeros(options['hiddenDimSize'])

    params['W_logistic'] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, 1))
    params['b_logistic'] = np.zeros((1,), dtype=config.floatX)

    return params



def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.iteritems():
        if key == 'W_emb': continue#####################
        tparams[key] = theano.shared(value, name=key)
    return tparams

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)
    return proj

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def build_model(tparams, options, Wemb):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(np.float32(0.))

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['embDimSize']])
    proj = rnn_layer(tparams, emb, options, mask=mask)

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    proj = proj[-1]

    # Compute the cost for the RNN
    cost = T.nnet.categorical_crossentropy(proj, y).mean()

    return use_noise, x, mask, y, proj, cost


def load_data(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(0.10 * dataSize)
    nValid = int(0.10 * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest+nValid]
    train_indices = ind[nTest+nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set

def adadelta(tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update

def calculate_auc(test_model, datasets):
    batchSize = 10
    n_batches = int(np.ceil(float(len(datasets[0])) / float(batchSize)))
    scoreVec = []
    for index in xrange(n_batches):
        x, mask = padMatrix(datasets[0][index*batchSize: (index+1)*batchSize])
        scoreVec.extend(list(test_model(x, mask)))
    labels = datasets[1]
    auc = roc_auc_score(list(labels), list(scoreVec))
    return auc

def padMatrix(seqs):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype(config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = np.array([ord(c) for c in unicode(s)], dtype='int32')

        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask


def rnn_layer(tparams, emb, options, mask=None):
    hiddenDimSize = options['hiddenDimSize']
    timesteps = emb.shape[0]
    if emb.ndim == 3: n_samples = emb.shape[1]
    else: n_samples = 1

    def stepFn(stepMask, e, h):
        h_new = T.tanh(T.dot(e, tparams['W_rnn']) + T.dot(h, tparams['U_rnn']) + tparams['b_rnn'])
        h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
        return h_new

    results, updates = theano.scan(fn=stepFn, sequences=[mask, emb], outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize), name='rnn_layer', n_steps=timesteps)

    return results[-1] 

def train_RNN_SVM(
    dataFile='data.txt',
    labelFile='label.txt',
    embFile='emb.txt',
    outFile='out.txt',
    inputDimSize=100,
    embDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    L2_reg=0.,
    batchSize=100,
    use_dropout=True
):
    options = locals().copy()
    
    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print('done!!')

    print('Building the model ... ')
    params = init_params(options)
    tparams = init_tparams(params)
    Wemb = theano.shared(params['W_emb'], name='W_emb')
    use_noise, x, mask, y, rnn_output, cost = build_model(tparams, options, Wemb)
    print('done!!')

    # Extract features from training, validation, and test sets
    rnn_output_function = theano.function(inputs=[x, mask], outputs=rnn_output, name='rnn_output_function')

    train_features = []
    for seq in trainSet[0]:
        x, mask = padMatrix([seq])
        train_features.append(rnn_output_function(x, mask).flatten())
    train_features = np.array(train_features)

    valid_features = []
    for seq in validSet[0]:
        x, mask = padMatrix([seq])
        valid_features.append(rnn_output_function(x, mask).flatten())
    valid_features = np.array(valid_features)

    test_features = []
    for seq in testSet[0]:
        x, mask = padMatrix([seq])
        test_features.append(rnn_output_function(x, mask).flatten())
    test_features = np.array(test_features)

    # Train SVM on extracted features
    svm = SVC(kernel='linear')
    svm.fit(train_features, trainSet[1])

    # Evaluate SVM on validation and test sets
    valid_predictions = svm.predict(valid_features)
    test_predictions = svm.predict(test_features)

    valid_auc = roc_auc_score(validSet[1], valid_predictions)
    test_auc = roc_auc_score(testSet[1], test_predictions)

    print('Validation AUC-ROC:', valid_auc)
    print('Test AUC-ROC:', test_auc)

    with open(outFile, 'w') as f:
        f.write('Validation AUC-ROC: {}\n'.format(valid_auc))
        f.write('Test AUC-ROC: {}\n'.format(test_auc))

        

if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    embFile = sys.argv[3]
    outFile = sys.argv[4]

    inputDimSize = 100 #The number of unique medical codes
    embDimSize = 100
    hiddenDimSize = 100 
    max_epochs = 100 #Maximum epochs to train
    L2_reg = 0.001 #L2 regularization for the logistic weight
    batchSize = 10 #The size of the mini-batch
    use_dropout = True 

    train_RNN_SVM(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, embDimSize=embDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, L2_reg=L2_reg, batchSize=batchSize, use_dropout=use_dropout)