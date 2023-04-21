import pickle

# Read the data from the text file
with open('seqs_data.txt', 'r') as text_file:
    lines = text_file.readlines()

# Convert the data into a suitable Python object (e.g., a list)
data = [line.strip() for line in lines]

# Use the pickle module to save the Python object to a file, using protocol version 2
with open('new_seqs_pickled.pkl', 'wb') as pickled_file:
    pickle.dump(data, pickled_file, protocol=2)
