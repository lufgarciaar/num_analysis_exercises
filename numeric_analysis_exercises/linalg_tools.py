import numpy as np

def lin_sol(augmented_matrix):
    
    matrix = augmented_matrix[:, 0:-1]
    
    rank = np.linalg.matrix_rank(matrix)
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)
    
    if (rank < rank_augmented):
        return 'no solution'

    unique_solution = rank == rank_augmented == matrix.shape[1]
    
    if unique_solution:
        return 'unique solution'
    
    else:
        return 'infinite solutions'

def diagonalize(A):
    eigenvalues_of_A, eigenvectors_of_A = np.linalg.eig(A)    
    diagonal_matrix = np.diagflat(eigenvalues_of_A)
    return diagonal_matrix, eigenvectors_of_A