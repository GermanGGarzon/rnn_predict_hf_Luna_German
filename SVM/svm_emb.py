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


def train_SVM(
    dataFile='data.txt',
    labelFile='label.txt',
    embFile='embFile.pkl',
    outFile='out.txt',
    inputDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    batchSize=100
):
    options = locals().copy()

    # Load the pickled embeddings
    print('Loading Embedded File')
    with open(embFile, 'rb') as f:
        embeddings = pickle.load(f)
    print("Embeddings shape:", np.array(embeddings).shape)

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


    # Train SVM on the embeddings
    print('Training SVM...')
    svm = SVC(kernel='linear', gamma=1000, degree=5, coef0=-0.5, C=0.001, probability=True)
    svm.fit(X_train, y_train_labels)  
    print('done!!')

    valid_auc = roc_auc_score(y_valid_labels, svm.predict_proba(X_valid)[:, 1])
    print('Validation AUC-ROC: {:.4f}'.format(valid_auc))
    test_auc = roc_auc_score(y_test_labels, svm.predict_proba(X_test)[:, 1])
    print('Test AUC-ROC: {:.4f}'.format(test_auc))

    with open(outFile, 'w') as f:
        f.write('Validation AUC-ROC: {:.4f}\n'.format(valid_auc))
        f.write('Test AUC-ROC: {:.4f}\n'.format(test_auc))


if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    embFile = sys.argv[3]
    outFile = sys.argv[4]

    inputDimSize = 1000 
    hiddenDimSize = 100 
    max_epochs = 100 
    batchSize = 1000 
    

    train_SVM(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, batchSize=batchSize)