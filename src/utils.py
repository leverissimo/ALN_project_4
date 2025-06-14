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


# =======================================
# ============ Question 1 ===============
# =======================================

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

def test_norma2_das_colunas():

    valores_possiveis_linha =  [1, 100, 1000, 10000]
    valores_possiveis_coluna =  [1, 100, 1000, 10000]
    
    for linha in valores_possiveis_linha:
        for col in valores_possiveis_coluna:
    
            A = generate_gaussian_matrix(linha, col)
            data = norma2_das_colunas(A)
            # hist, bin_edges = make_Histogram(data, bins=30)
            title = f"Normas 2 das Colunas - Matriz {linha}X{col}"
            # plot_histogram(hist, bin_edges, title=title, xlabel='Norma 2', ylabel='Frequência')
            plot_histogram_seaborn(data, bins=25, title=title, xlabel='Norma 2',  ylabel='Frequência')
# =======================================
# ============ Question 2 ===============
# =======================================


def produto_interno(A):
    """Calcula o produto interno entre todas as colunas de uma matriz.
    
    Args:
        A (np.ndarray): Matriz com as colunas a serem calculadas.
    
    Returns:
        list: Lista com os produtos internos entre todas as colunas.
    """
    resultados = []
    
    for i in range(A.shape[1]):
        for j in range(i+1, A.shape[1]):
            inter_prod = np.dot(A[:, i], A[:, j])
            resultados.append(inter_prod)
    return resultados
        
def test_produto_interno():
    """
    Testa a função produto_interno gerando matrizes Gaussianas de diferentes
    dimensões de coluna e plota histogramas dos produtos internos calculados.

    Para cada dimensão de coluna especificada em 'intervalos', a função:
    1. Gera uma matriz Gaussiana com 100 linhas e 'i' colunas.
    2. Calcula o produto interno entre todas as colunas da matriz.
    3. Cria um histograma dos produtos internos calculados.
    4. Plota e salva o histograma em um arquivo na pasta 'figures/produto_interno'.

    """

    intervalos = [100, 200, 500, 1000]
    for i in intervalos:
        A = generate_gaussian_matrix(100, i)
        resultados = produto_interno(A)
        hist, bin_edges = make_Histogram(resultados, bins=30)
        title = f"Histograma do Produto Interno - Dimensão {i}"
        # plot_histogram(hist, bin_edges, title=title, xlabel='Produto Interno', ylabel='Frequência', folder='figures/produto_interno')
        plot_histogram_seaborn(data=resultados, bins =25,xlabel='Produto Interno', title=title, ylabel='Frequência', folder='figures/produto_interno' )

if __name__ == "__main__":
    # test_norma2_das_colunas(5)
    test_norma2_das_colunas()
    test_produto_interno()
    
    print("Teste concluído.")