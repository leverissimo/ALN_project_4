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
    data = []

    for i in range(A.shape[1]):
        data.append(np.linalg.norm(A[:, i]))

    return data

def make_Histogram(data, bins=20):
    """Cria um histograma a partir dos dados fornecidos.

    Args:
        data (np.ndarray): Dados para o histograma.
        bins (int): Número de bins do histograma.

    Returns:
        tuple: Valores do histograma e os limites dos bins.
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

def plot_histogram(hist, bin_edges, title='Histograma', xlabel='Valor', ylabel='Frequência'):
    """Plota um histograma.

    Args:
        hist (np.ndarray): Valores do histograma.
        bin_edges (np.ndarray): Limites dos bins.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo x.
        ylabel (str): Rótulo do eixo y.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def test_norma2_das_colunas(n):
    for i in range(n):
        A = generate_gaussian_matrix(10000, 10000)
        data = norma2_das_colunas(A)
        hist, bin_edges = make_Histogram(data, bins=30)
        plot_histogram(hist, bin_edges, title='Histograma das Normas 2 das Colunas', xlabel='Norma 2', ylabel='Frequência')