import pandas as pd
import pickle


csv_file_path = 'ADMISSIONS.csv'
data = pd.read_csv(csv_file_path)


pickle_file_path = 'seqs_pickled.pkl'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(data, file)


with open(pickle_file_path, 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)
