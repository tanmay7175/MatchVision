import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def match_embeddings(file1, file2):
    data1 = np.load(file1, allow_pickle=True).item()
    data2 = np.load(file2, allow_pickle=True).item()

    sim = cosine_similarity(data1["embeddings"], data2["embeddings"])
    matches = sim.argmax(axis=1)

    matched = []
    for idx1, idx2 in enumerate(matches):
        matched.append((data1["files"][idx1], data2["files"][idx2]))
    
    return matched
