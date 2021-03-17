import numpy as np

def linear_solutions(augmented_matrix):
    
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
