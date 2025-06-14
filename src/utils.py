import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_histogram(hist, bin_edges, title='Histograma', xlabel='Valor', ylabel='Frequência', folder='figures/histograms'):
    """Plota um histograma.

    Args:
        hist (np.ndarray): Valores do histograma.
        bin_edges (np.ndarray): Limites dos bins.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo x.
        ylabel (str): Rótulo do eixo y.
    """
    os.makedirs(folder, exist_ok=True)

    filename = title.lower().replace(' ', '_')
    filepath = os.path.join(folder, f"{filename}.png")
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico salvo em: {filepath}")    



def plot_histogram_seaborn(data, bins=20, title='Histograma', xlabel='Valor', ylabel='Frequência', folder='figures/histograms'):
    os.makedirs(folder, exist_ok=True)

    filename = title.lower().replace(' ', '_')
    filepath = os.path.join(folder, f"{filename}.png")

    plt.figure(figsize=(10, 6))
    # sns.set(style="darkgrid")  # ou "whitegrid", "ticks"...
    sns.histplot(data, bins=bins, kde=True, stat='density',  color='skyblue', edgecolor='black')
    sns.kdeplot(data, color="purple", linewidth=2, label="KDE (Densidade)")
    # sns.kdeplot(data, color="red
    # ", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend() 
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico salvo em: {filepath}")

def test_norma2_das_colunas(n):
    for i in range(n):
        A = generate_gaussian_matrix(1000, 1000)
        data = norma2_das_colunas(A)
        # hist, bin_edges = make_Histogram(data, bins=30)
        title = f"Histograma das Normas 2 das Colunas - Execução {i+1}"
        # plot_histogram(hist, bin_edges, title=title, xlabel='Norma 2', ylabel='Frequência')
        plot_histogram_seaborn(data, bins=25, title=title, xlabel='Norma 2',  ylabel='Frequência')
if __name__ == "__main__":
    test_norma2_das_colunas(5)
    print("Teste concluído.")