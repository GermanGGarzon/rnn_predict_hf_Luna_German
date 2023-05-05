# Heart Failure Prediction using RNN - Reproduction

# Requirement
For the reproduction effort, we reproduced the code for GRU and developed our own implementation of KNN and SVM. All of these require the use of an Anaconda environment with Python 3.5, numpy, scikit-learn, theano, and torchvision.

Open a command prompt and enter the following commands:
- conda create --name py3 python=3.5
- activate py3
- conda install numpy scikit-learn theano torchvision
- Once all the packages have been installed, move on to the section labeled "How to execute" below


# Pickled Files
We have a set of pickled files that we run with our implementation. These pickled files are created with a python script we developed. The scripts need to be opened and modified to indicate the input and output files. The scripts are in Data from MediSyn/ICU-Synthetic/ICU-Synthetic/.

In order to create the embiddings files we generated them with a python script we developed in the Create Embeddings folder. This python file also needs to be opened and modified to specify the embeddings range, size of the embeddings and output file name. The files have already been generated and are named embeddings.pkl

# How to execute
The GRU, SVM, and KNN folders contains the required python files to run GRU, SVM, and KNN respectively. cd into the directory of your choosing.

To Run svm_onehot.py and svm_emb.py use:
- python svm_onehot.py new_seqs.pkl new_labels.pkl outfile
- python svm_emb.py new_seqs.pkl new_labels.pkl embeddings.pkl outfile

To Run knn_onehot.py and knn_emb.py use:
- python knn_onehot.py new_seqs.pkl new_labels.pkl outfile
- python knn_emb.py new_seqs.pkl new_labels.pkl embeddings.pkl outfile

To Run gru_onehot.py and gru_emb.py use:
- python gru_onehot.py new_seqs.pkl new_labels.pkl outfile
- python gru_emb.py new_seqs.pkl new_labels.pkl embeddings.pkl outfile


# Heart Failure Prediction using RNN - Original
This is a simple RNN (implemented with Gated Recurrent Units) for predicting a HF diagnosis given patient records.
There are four different versions. The code provided by the authors of the original paper can be found in the main directory of the repo: /rnn_predict_hf_Luna_German/
 

1. gru_onehot.py: This uses one-hot encoding for the medical code embedding
2. gru_onehot_time.py: This uses one-hot encoding for the medical code embedding. This uses time information in addition to the code sequences
3. gru_emb.py: This uses pre-trained medical code embeddings. 
4. gru_emb_time.py: This uses pre-trained medical code embeddings. This suses time information in addition to the code sequences.

The data are synthetic and make no sense at all. It is intended only for testing the codes.

1. sequences.pkl: This is a pickled list of list of integers. Each integer is assumed to be some medical code.
2. times.pkl: This is a pickled list of list of integers. Each integer is assumed to the time at which the medical code occurred.
3. labels.pkl: This is a pickled list of 0 and 1s.
4. emb.pkl: This is a randomly generated code embedding of size 100 X 100

# Requirement
In order to run the existing code from the authors (GRU), only an Anaconda environment with Python 2.7, numpy, scikit-learn, and theano are required.

# How to Execute
1. python gru_onehot.py sequences.pkl labels.pkl <output>
2. python gru_onehot_time.py sequences.pkl times.pkl labels.pkl <output>
3. python gru_emb.py sequences.pkl labels.pkl emb.pkl <output>
4. python gru_emb_time.py sequences.pkl times.pkl labels.pkl emb.pkl <output>

All scripts will divide the data into training set, validation set, and test set. They will run for a fixed number of epochs. At each epoch, "Validation AUC" will be calculated using the validation set, and if it is the best "Validation AUC" so far, the test set will be used to calculate "Test AUC". The model with the best "Test AUC" will be saved at the end of the training.
