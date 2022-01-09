import numpy as np

def scale(matrix):
    new_matrix = []
    for row in matrix.T:
        if np.std(row) == 0.0:
            continue
        else:
            new_matrix.append(np.array((row - np.mean(row,axis=0)) / np.std(row)))
    new_matrix = np.array(new_matrix)
    return new_matrix
