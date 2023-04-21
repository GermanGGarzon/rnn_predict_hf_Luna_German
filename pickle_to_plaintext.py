import pickle

with open('seqs.pkl', 'rb') as pickled_file:
    data = pickle.load(pickled_file)


with open('seqs_data.txt', 'w') as plaintext_file:
    for item in data:
        # Assuming data is a list
        # You can customize how you want to format the data in the plaintext file
        plaintext_file.write('{}\n'.format(item))


