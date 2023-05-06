import sys, random
import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split
import ast

import torch
import torch.nn as nn
import torch.optim as optim

theano.config.cxx = ""


SEED = int(time.time())

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

    params['W_emb'] = np.random.uniform(-0.01, 0.01, (inputDimSize, hiddenDimSize)).astype(config.floatX)

    # Use Glorot initialization for W_rnn and U_rnn
    bound = np.sqrt(6. / (options['inputDimSize'] + options['hiddenDimSize']))
    params['W_rnn'] = np.random.uniform(-bound, bound, (options['inputDimSize'], options['hiddenDimSize']))
    bound = np.sqrt(6. / (options['hiddenDimSize'] + options['hiddenDimSize']))
    params['U_rnn'] = np.random.uniform(-bound, bound, (options['hiddenDimSize'], options['hiddenDimSize']))

    params['b_rnn'] = np.zeros(options['hiddenDimSize'])

    params['W_logistic'] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, 1))
    params['b_logistic'] = np.zeros((1,), dtype=config.floatX)

    return params



def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.items():
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
	trng = RandomStreams(123)
	use_noise = theano.shared(numpy_floatX(0.))

	x = T.matrix('x', dtype='int32')
	mask = T.matrix('mask', dtype=config.floatX)
	y = T.vector('y', dtype='int32')

	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	emb = Wemb[x.flatten()].reshape([n_timesteps,n_samples,options['embDimSize']])

	proj = rnn_layer(tparams, emb, options, mask=mask)
	if options['use_dropout']: proj = dropout_layer(proj, use_noise, trng)

	p_y_given_x = T.nnet.sigmoid(T.dot(proj, tparams['W_logistic']) + tparams['b_logistic'])
	L = -(y * T.flatten(T.log(p_y_given_x)) + (1 - y) * T.flatten(T.log(1 - p_y_given_x)))
	cost = T.mean(L)

	if options['L2_reg'] > 0.: cost += options['L2_reg'] * (tparams['W_logistic'] ** 2).sum()

	return use_noise, x, mask, y, p_y_given_x, cost


def load_data(dataFile, labelFile, test_size=0.2, valid_size=0.2):
    sequences = np.array(pickle.load(open(dataFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))

    data = np.column_stack((sequences, labels))
    np.random.shuffle(data)

    sequences_shuffled = data[:, 0]
    labels_shuffled = data[:, 1]

    X_train, X_temp, y_train, y_temp = train_test_split(sequences_shuffled, labels_shuffled, test_size=test_size + valid_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + valid_size))

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

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


max_sequence_length = 100

def padMatrix(seqs, max_len):
    lengths = [len(ast.literal_eval(s)) for s in seqs]
    n_samples = len(seqs)

    x = np.zeros((n_samples, max_len, inputDimSize), dtype='float32')


    for idx, seq_str in enumerate(seqs):
        seq = ast.literal_eval(seq_str)
        seq_length = lengths[idx]

        one_hot_seq = np.zeros((len(seq), inputDimSize))
        for i, char in enumerate(seq):
            one_hot_seq[i, int(char)] = 1

        one_hot_seq = one_hot_seq[:max_len]
        
        x[idx, :min(seq_length, max_len), :] = one_hot_seq[:min(seq_length, max_len), :]


    return x



class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_prob=0.5, bidirectional=True):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x


def train_RNN(
    dataFile='data.txt',
    labelFile='label.txt',
    outFile='out.txt',
    inputDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    lr=0.001,
    batchSize=100,
    dropout_prob=0.5,
    L2_reg=1e-4,
    num_layers=2,  
    bidirectional=True
):
    options = locals().copy()
    bestValidAuc = 0.
    bestTestAuc = 0.

    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    y_train_labels = np.array(trainSet[1], dtype=np.float32)
    y_valid_labels = np.array(validSet[1], dtype=np.float32)
    y_test_labels = np.array(testSet[1], dtype=np.float32)
    print('done!!')
    
    max_len_train = np.min([np.max([len(s) for s in trainSet[0]]), max_sequence_length])

    rnn_model = RNNModel(inputDimSize, hiddenDimSize, num_layers=num_layers, dropout_prob=dropout_prob, bidirectional=bidirectional)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)


    def train_epoch():
        rnn_model.train()
        total_loss = 0
        for i in range(0, len(trainSet[0]), batchSize):
            X_batch_padded = padMatrix(trainSet[0][i:i + batchSize],max_len_train)
            X_batch_tensor = torch.tensor(X_batch_padded, dtype=torch.float32)
            y_batch_tensor = torch.tensor(y_train_labels[i:i + batchSize], dtype=torch.float32)

            optimizer.zero_grad()
            outputs = rnn_model(X_batch_tensor).squeeze()
            loss = criterion(outputs, y_batch_tensor)
            
            l2_reg = torch.tensor(0., dtype=torch.float32)
            for param in rnn_model.parameters():
                l2_reg += torch.norm(param)**2
            loss += L2_reg * l2_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(trainSet[0])

    def calculate_auc(dataset):
        rnn_model.eval()
        X_data_padded = padMatrix(dataset[0],max_len_train)
        X_data_tensor = torch.tensor(X_data_padded, dtype=torch.float32)
        with torch.no_grad():
            outputs = rnn_model(X_data_tensor).squeeze()
        y_pred = torch.sigmoid(outputs).numpy()
        return roc_auc_score(dataset[1], y_pred)



    # Add early stopping
    patience = 15
    epochs_no_improve = 0
    early_stopping = False

    for epoch in range(max_epochs):
        train_loss = train_epoch()
        valid_auc = calculate_auc(validSet)

        # Update learning rate
        scheduler.step(valid_auc)

        if (valid_auc > bestValidAuc):
            bestValidAuc = valid_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                early_stopping = True

        print('Epoch {:3d}, Loss: {:.4f}, Validation AUC-ROC: {:.4f}'.format(epoch + 1, train_loss, valid_auc))

        test_auc = calculate_auc(testSet)
        if (test_auc > bestTestAuc):
            bestTestAuc = test_auc

        print('Test AUC-ROC: {:.4f}'.format(test_auc))

        if early_stopping:
            print('Early stopping triggered')
            break

        print('Best Validation score: {:.4f}'.format(bestValidAuc))
        print('Best Test score: {:.4f}'.format(bestTestAuc))
        print('\n')

if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    outFile = sys.argv[3]

    inputDimSize = 1000 
    hiddenDimSize = 100 
    max_epochs = 50 
    batchSize = 10 
    dropout_prob=0.7
    lr = 0.01 
    L2_reg = 1e-4
    

    train_RNN(dataFile=dataFile, labelFile=labelFile, outFile=outFile, inputDimSize=inputDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, lr=lr, batchSize=batchSize, dropout_prob=dropout_prob,L2_reg=L2_reg)