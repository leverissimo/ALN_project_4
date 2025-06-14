import numpy as np


def generate_gaussian_matrix(M: int, N:int)->np.ndarray:
    """Cria uma matriz Gaussiana com as entradas
    seguindo uma distribuição normal com media 0 e 
    variancia igual a 1 

    Args:
        M (int): Dimensão das linhas da Matriz
        N (int): Dimensão das colunas da Matriz

    Returns:
        np.ndarray: Matriz Gaussiana
    """
    
    matriz = np.random.normal(loc=0, scale = 1, size=(M, N))
    
    return matriz

def norma2_das_colunas(A):
    A = A.copy()
    norm = []

    for i in range(A.shape[1]):
        norm.append(np.linalg.norm(A[:, i]))

    return norm
