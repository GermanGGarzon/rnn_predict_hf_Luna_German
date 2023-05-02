import numpy as np
import cPickle as pickle

embeddings = np.random.uniform(low=-10, high=10, size=(1000, 1000))

with open("embeddings_pklv2.txt", "w") as f:
    for row in embeddings:
        line = " ".join(str(x) for x in row)
        f.write(line + "\n")


with open("embeddings_pklv2.txt", "r") as f:
    loaded_embeddings = np.array([list(map(float, line.strip().split())) for line in f])

with open("embeddings_pklv2.pkl", "wb") as f:
    pickle.dump(loaded_embeddings, f)
