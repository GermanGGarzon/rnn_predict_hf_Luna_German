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

def predict(model, X):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs).squeeze()
        scores = torch.sigmoid(outputs)  # Convert logits to probabilities
    return scores.numpy()

# Define the PyTorch RNN model
class RNNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.rnn.input_size)  # Reshape the input tensor
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x


def train_RNN(
    dataFile='data.txt',
    labelFile='label.txt',
    embFile='embeddings.plk',
    outFile='out.txt',
    inputDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    lr=0.001,
    batchSize=100,
    dropout_prob=0.5,
    L2_reg=1e-4
):
    options = locals().copy()
    bestValidAuc = 0.
    bestTestAuc = 0.
    
    print('Loading Embedded File')
    with open(embFile, 'rb') as f:
        embeddings = pickle.load(f)
    print('done!!')

    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    y_train_labels = np.array(trainSet[1], dtype=np.int64)
    y_valid_labels = np.array(validSet[1], dtype=np.int64)
    y_test_labels = np.array(testSet[1], dtype=np.int64)
    print('done!!')
    

    train_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in trainSet[0]])
    valid_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in validSet[0]])
    test_indices = np.array([np.array(ast.literal_eval(seq), dtype=np.int64) for seq in testSet[0]])

 
    X_train = padMatrix([embeddings[seq] for seq in train_indices])
    X_valid = padMatrix([embeddings[seq] for seq in valid_indices])
    X_test = padMatrix([embeddings[seq] for seq in test_indices])

    rnn_model = RNNModel(inputDimSize, hiddenDimSize, dropout_prob)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=lr)

    def train_epoch():
        rnn_model.train()
        total_loss = 0
        for i in range(0, len(trainSet[0]), batchSize):
            X_batch_padded = X_train[i:i + batchSize]
            X_batch_tensor = torch.tensor(X_batch_padded, dtype=torch.float32)
            y_batch_tensor = torch.tensor(y_train_labels[i:i + batchSize], dtype=torch.float32)  # Add this line back

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

    for epoch in range(max_epochs):
        train_loss = train_epoch()
        valid_outputs = rnn_model(torch.tensor(X_valid, dtype=torch.float32)).squeeze()
        
        valid_auc = roc_auc_score(y_valid_labels, valid_outputs.detach().numpy()) 
        if (valid_auc > bestValidAuc):
            bestValidAuc = valid_auc
            print('Best validation score: {:.4f}'.format(valid_auc))
        print('Epoch {:3d}, Loss: {:.4f}, Validation AUC-ROC: {:.4f}'.format(epoch + 1, train_loss, valid_auc))

        test_scores = predict(rnn_model, X_test)
        test_auc = roc_auc_score(y_test_labels, test_scores)
        if (test_auc > bestTestAuc):
            bestTestAuc = test_auc
            print('Best Test score: {:.4f}'.format(test_auc))

        print('Test AUC-ROC: {:.4f}'.format(test_auc))
        print('\n')
        
    print('Best Validation score: {:.4f}'.format(bestValidAuc))
    print('Best Test score: {:.4f}'.format(bestTestAuc))

if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    embFile = sys.argv[3]
    outFile = sys.argv[4]

    inputDimSize = 1000 
    hiddenDimSize = 1000 
    max_epochs = 100 
    lr = 0.01 
    batchSize = 100 
    dropout_prob=0.7
    L2_reg = 1e-4
    

    train_RNN(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, lr=lr, batchSize=batchSize, dropout_prob=dropout_prob,L2_reg=L2_reg)