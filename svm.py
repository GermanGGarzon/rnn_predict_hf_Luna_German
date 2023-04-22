import sys
import numpy as np
import pickle
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

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

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    return train_set, valid_set, test_set

def calculate_auc(svm_model, datasets):
    scoreVec = svm_model.decision_function(datasets[0])
    labels = datasets[1]
    auc = roc_auc_score(list(labels), list(scoreVec))
    return auc

def train_and_evaluate_SVM(trainSet, validSet, testSet, modelFile='out.txt', kernel='rbf', C=1.0, max_iter=-1):

    print('Building the model ... ')
    svm_model = SVC(kernel=kernel, C=C, max_iter=max_iter, probability=True)
    print('done!')

    print('Training the model ... ')
    print(trainSet[0].shape)
    print(trainSet[1].shape)
    svm_model.fit(np.vstack(trainSet[0]), trainSet[1])
    print('done!')

    print('Evaluating the model ... ')
    valid_auc = calculate_auc(svm_model, validSet)
    test_auc = calculate_auc(svm_model, testSet)
    print('done!')

    print('Validation AUC: {}'.format(valid_auc))
    print('Test AUC:: {}'.format(test_auc))

    print('Saving the model ... ')
    with open(modelFile, 'wb') as f:
        pickle.dump(svm_model, f)
    print('done!')


if __name__ == '__main__':
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    outFile = sys.argv[3]

    kernel = 'rbf'
    C = 1.0
    max_iter = -1
    
    trainSet, validSet, testSet = load_data(dataFile, labelFile)
    train_and_evaluate_SVM(trainSet, validSet, testSet, modelFile=outFile)

