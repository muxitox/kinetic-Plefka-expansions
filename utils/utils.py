import numpy as np

def broadcast_columns(vec, size):
    '''
    Creates a matrix where each column is a copy of the vector vec
    '''
    return np.einsum('j,k->jk', vec, np.ones(size))

def broadcast_rows(vec, size):
    """
    Creates a matrix where each row is a copy of the vector vec
    """
    return np.einsum('j,k->kj', vec, np.ones(size))