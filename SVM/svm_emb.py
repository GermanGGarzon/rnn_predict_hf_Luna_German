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

def calculate_auc(svm_model, rnn_model, dataset):
    # Extract features using the RNN model
    X_data_padded, _ = padMatrix(dataset[0])
    X_data_tensor = torch.tensor(X_data_padded, dtype=torch.float32).permute(1, 0, 2)
    print("X_data_tensor shape:", X_data_tensor.shape)

    with torch.no_grad():
        X_data_tensor = X_data_tensor.mean(dim=-1)
        data_features = rnn_model(X_data_tensor).numpy()

    # Make predictions using the trained SVM model
    y_pred = svm_model.decision_function(data_features.reshape(1, -1))


    # Calculate the AUC-ROC score
    auc_score = roc_auc_score(dataset[1], y_pred)

    return auc_score



max_sequence_length = 100

def padMatrix(seqs):
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    max_len = np.min([np.max(lengths), max_sequence_length])

    x = np.zeros((max_len, n_samples, inputDimSize), dtype='float32')
    x_mask = np.zeros((max_len, n_samples), dtype='float32')

    for idx, s in enumerate(seqs):
        seq_length = lengths[idx]
        one_hot_seq = np.zeros((max_len, inputDimSize), dtype='float32')
        for i, c in enumerate(s[:max_len]):
            one_hot_seq[i, ord(c)] = 1
        x[:, idx, :] = one_hot_seq
        x_mask[:min(seq_length, max_len), idx] = 1.

    return x, x_mask

def load_embeddings(embedding_file):
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    max_length = max([len(e.split()) for e in embeddings])
    padded_embeddings = []
    
    for e in embeddings:
        emb_array = np.fromstring(e, sep=' ', dtype=np.float32)
        pad_length = max_length - len(emb_array)
        padded_emb_array = np.pad(emb_array, (0, pad_length), 'constant', constant_values=0)
        padded_embeddings.append(padded_emb_array)
    
    embeddings = np.stack(padded_embeddings)
    return embeddings


# Define the PyTorch RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, embeddings):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float32), freeze=True)
        self.proj = nn.Linear(embeddings.shape[1], input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Change this line to use nn.RNN
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x.long()).squeeze(dim=2)
        x = self.proj(x)

        # Process each sequence in the batch individually
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            sequence = x[i].unsqueeze(0)
            out, _ = self.rnn(sequence)
            out = self.fc(out[:, -1, :])
            outputs.append(out)

        # Stack the outputs to get the final tensor
        x = torch.stack(outputs, dim=0).squeeze()
        return x



def train_RNN_SVM(
    dataFile='data.txt',
    labelFile='label.txt',
    embFile='emb.txt',
    outFile='out.txt',
    inputDimSize=100,
    embDimSize=100,
    hiddenDimSize=100,
    max_epochs=100,
    lr=0.001,
    batchSize=100,
    use_dropout=True
):
    options = locals().copy()
    
    print('Loading Embedded File')
    embeddings = load_embeddings(embFile)
    print("Embeddings shape:", np.array(embeddings).shape)


    
    print('Loading data ... ')
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print('done!!')

    print('Building the model ... ')
    #params = init_params(options)
    #tparams = init_tparams(params)
    #Wemb = theano.shared(params['W_emb'], name='W_emb')
    #use_noise, x, mask, y, rnn_output, cost = build_model(tparams, options, Wemb)
    
    # Extract features from training, validation, and test sets
    rnn = RNNModel(inputDimSize, hiddenDimSize, embeddings)

    # Train the RNN model
    y_train_array = np.array(trainSet[1])
    X_train_padded, _ = padMatrix(trainSet[0])
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).permute(1, 0, 2)
    X_train_tensor = X_train_tensor.mean(dim=-1)
    y_train_tensor = torch.tensor(y_train_array.astype(np.float32)).unsqueeze(1)
    print('done!!')


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = rnn(X_train_tensor[:, :100])
        loss = criterion(outputs.unsqueeze(-1), y_train_tensor)
        loss.backward()
        optimizer.step()

    # Extract features from the RNN model
    with torch.no_grad():
        train_features = rnn(X_train_tensor).numpy()

    # Train SVM on extracted features
    print('Training SVM...')
    train_features = np.reshape(train_features, (train_features.shape[0], -1))
    svm = SVC(kernel='linear', gamma=1000, degree=5, coef0=-0.5, C=0.001)
    svm.fit(train_features, trainSet[1])  # Pass extracted features and corresponding labels
    print('done!!')

    valid_auc = calculate_auc(svm, rnn, validSet)
    
    test_auc = calculate_auc(svm, rnn, testSet)

    print('Validation AUC-ROC: {:.4f}'.format(valid_auc))
    print('Test AUC-ROC: {:.4f}'.format(test_auc))

    with open(outFile, 'w') as f:
        f.write('Validation AUC-ROC: {:.4f}\n'.format(valid_auc))
        f.write('Test AUC-ROC: {:.4f}\n'.format(test_auc))

if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    embFile = sys.argv[3]
    outFile = sys.argv[4]

    inputDimSize = 100 #The number of unique medical codes
    embDimSize = 100
    hiddenDimSize = 100 
    max_epochs = 100 #Maximum epochs to train
    lr = 0.001 
    batchSize = 10 #The size of the mini-batch
    use_dropout = True 
    

    train_RNN_SVM(dataFile=dataFile, labelFile=labelFile, embFile=embFile, outFile=outFile, inputDimSize=inputDimSize, embDimSize=embDimSize, hiddenDimSize=hiddenDimSize, max_epochs=max_epochs, lr=lr, batchSize=batchSize, use_dropout=use_dropout)