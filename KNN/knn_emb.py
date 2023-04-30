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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import ast

import torch
import torch.nn as nn
import torch.optim as optim

theano.config.cxx = ""


def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.iteritems():
        new_params[key] = value.get_value()
    return new_params

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)

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

    emb = Wemb[x.flatten()].reshape([n_timesteps, n_samples, options['inputDimSize']])

    # Apply a linear transformation to the embeddings
    h = T.dot(emb, tparams['W']) + tparams['b']
    
    # Apply the sigmoid function to compute the output probabilities
    p_y_given_x = T.nnet.sigmoid(h)

    # Calculate the cost using binary cross-entropy
    L = -(y * T.flatten(T.log(p_y_given_x)) + (1 - y) * T.flatten(T.log(1 - p_y_given_x)))
    cost = T.mean(L)

    if options['L2_reg'] > 0.:
        cost += options['L2_reg'] * (tparams['W'] ** 2).sum()

    return use_noise, x, mask, y, p_y_given_x, cost



def load_data(dataFile, labelFile, test_size=0.2, valid_size=0.2):
    sequences = np.array(pickle.load(open(dataFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))

    # Shuffle the data and labels together
    data = np.column_stack((sequences, labels))
    np.random.shuffle(data)

    sequences_shuffled = data[:, 0]
    labels_shuffled = data[:, 1]

    # Split the data into train, validation, and test sets
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

def calculate_auc(knn_model, rnn_model, dataset, use_rnn=True):
    X_data_padded, _ = padMatrix(dataset[0])

    if use_rnn:
        X_data_tensor = torch.tensor(X_data_padded, dtype=torch.float32).permute(1, 0, 2)
        with torch.no_grad():
            data_features = rnn_model(X_data_tensor).numpy()
    else:
        data_features = np.reshape(X_data_padded, (X_data_padded.shape[1], -1))

    y_pred_proba = knn_model.predict_proba(data_features)
    y_pred = y_pred_proba[:, 1] # Get the probabilities of the positive class
    auc_score = roc_auc_score(dataset[1], y_pred)

    return auc_score




max_sequence_length = 100

def padMatrix(seqs):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    max_len = np.min([np.max(lengths), max_sequence_length])

    x = np.zeros((n_samples, max_len * inputDimSize), dtype='float32')

    for idx, s in enumerate(seqs):
        seq_length = lengths[idx]
        one_hot_seq = s[:max_len]
        x[idx, :min(seq_length, max_len) * inputDimSize] = one_hot_seq.flatten()[:min(seq_length, max_len) * inputDimSize]

    return x





def train_KNN(
    dataFile='data.txt',
    labelFile='label.txt',
    embFile='embFile.pkl',
    outFile='out.txt',
    inputDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    lr=0.001,
    batchSize=100,
    use_dropout=True
):
    options = locals().copy()

    # Load the pickled embeddings
    with open(embFile, 'rb') as f:
        embeddings = pickle.load(f)

    # Load the data
    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    y_train_labels = np.array(trainSet[1], dtype=np.int64)
    y_valid_labels = np.array(validSet[1], dtype=np.int64)
    y_test_labels = np.array(testSet[1], dtype=np.int64)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print('done!!')

    # Convert sequences to integer indices
    train_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in trainSet[0]])
    valid_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in validSet[0]])
    test_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in testSet[0]])

    # Get the embeddings for the training, validation, and test sets
    X_train = padMatrix([embeddings[seq] for seq in train_indices])
    X_valid = padMatrix([embeddings[seq] for seq in valid_indices])
    X_test = padMatrix([embeddings[seq] for seq in test_indices])



    # Train KNN on the embeddings
    print('Training KNN...')
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train_labels)
    print('done!!')

    valid_auc = roc_auc_score(y_valid_labels, knn.predict_proba(X_valid)[:, 1])
    print('Validation AUC-ROC: {:.4f}'.format(valid_auc))
    test_auc = roc_auc_score(y_test_labels, knn.predict_proba(X_test)[:, 1])
    print('Test AUC-ROC: {:.4f}'.format(test_auc))

    with open(outFile, 'w') as f:
        f.write('Validation AUC-ROC: {:.4f}\n'.format(valid_auc))
        f.write('Test AUC-ROC: {:.4f}\n'.format(test_auc))





if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    embFile = sys.argv[3]
    outFile = sys.argv[4]

    inputDimSize = 1000 #The number of unique medical codes
    hiddenDimSize = 1000 
    max_epochs = 100 #Maximum epochs to train
    lr = 0.01 
    batchSize = 1000 #The size of the mini-batch
    use_dropout = True 
    

    train_KNN(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, lr=lr, batchSize=batchSize, use_dropout=use_dropout)