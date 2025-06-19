import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

import warnings
from scipy.optimize import OptimizeWarning

# Ignorar warnings específicos
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

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

import scipy.stats as stats

def melhor_ajuste_distribuicao(data):
    # Lista de distribuições candidatas (incluindo Gama e Beta)
    distribuicoes = [
        ('norm', stats.norm),      # Normal (suporte: -inf a +inf)
        ('chi', stats.chi),        # Chi (suporte: x ≥ 0)
        ('chi2', stats.chi2),      # Chi-quadrado (suporte: x ≥ 0)
        ('rayleigh', stats.rayleigh),  # Rayleigh (suporte: x ≥ 0)
        ('gamma', stats.gamma),    # Gama (suporte: x > 0)
        ('beta', stats.beta)       # Beta (suporte: 0 ≤ x ≤ 1)
    ]
    
    melhor_p = -1
    melhor_nome = ''
    
    for nome, distrib in distribuicoes:
        try:
            # Ajuste especial para a distribuição Beta (requer dados em [0, 1])
            if nome == 'beta':
                # Normaliza os dados para o intervalo [0, 1]
                data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
                params = distrib.fit(data_normalized)
                ks_stat, p_valor = stats.kstest(data_normalized, nome, args=params)
            else:
                params = distrib.fit(data)
                ks_stat, p_valor = stats.kstest(data, nome, args=params)
            
            if p_valor > melhor_p:
                melhor_p = p_valor
                melhor_nome = nome
        except Exception as e:
            print(f"Erro ao ajustar {nome}: {str(e)}")
            continue
            
    return melhor_nome

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



def plot_histogram_seaborn(data, bins=20, title='Histograma', xlabel='Valor', ylabel='Frequência', folder='figures/normas'):
    os.makedirs(folder, exist_ok=True)

    filename = title.lower().replace(' ', '_')
    filepath = os.path.join(folder, f"{filename}.png")
    mean = np.mean(data)
    std = np.std(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    plt.figure(figsize=(10, 6))
    # sns.set(style="darkgrid")  # ou "whitegrid", "ticks"...
    sns.histplot(data, bins=bins, kde=True, stat='density',  color='skyblue', edgecolor='black', ax =ax)
    sns.kdeplot(data, color="purple", linewidth=2, label="KDE (Densidade)", ax =ax)
    ax.axvline(mean,        ls='--', lw=2,  color='blue',   label=f'Média: {mean:.2f}')
    # sns.kdeplot(data, color="red
    # ", linewidth=2)
    # Títulos e eixos
    # ax.set_title(title)
    ax.set_title(f"{title} (σ = {std:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Agora o Matplotlib encontra todos os rótulos
    ax.legend()

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo em: {filepath}")


def test_norma2_das_colunas():

    valores_possiveis_linha =  [1, 100, 1000, 10000]
    valores_possiveis_coluna =  [1, 100, 1000, 10000]
    
    for linha in valores_possiveis_linha:
        for col in valores_possiveis_coluna:
    
            A = generate_gaussian_matrix(linha, col)
            data = norma2_das_colunas(A)
            melhor_ajuste = melhor_ajuste_distribuicao(data)
            print(f"Melhor Ajuste (Distibuição na Matriz {linha}X{col}: {melhor_ajuste} )")
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

    intervalos = [100, 200, 500, 1000, 5000, 10000]
    for i in intervalos:
        A = generate_gaussian_matrix(100, i)
        resultados = produto_interno(A)
        melhor_ajuste = melhor_ajuste_distribuicao(resultados)
        hist, bin_edges = make_Histogram(resultados, bins=30)
        print(f"Melhor ajuste na matriz 100 X {i}: {melhor_ajuste}")
        title = f"Produto Interno das Colunas - Matriz {i}"
        # plot_histogram(hist, bin_edges, title=title, xlabel='Produto Interno', ylabel='Frequência', folder='figures/produto_interno')
        plot_histogram_seaborn(data=resultados, bins =25,xlabel='Produto Interno', title=title, ylabel='Frequência', folder='figures/produto_interno' )
      
# =======================================
# ============= Question 3 ==============
# =======================================

def distribuicao_do_maximo(A):
    norm = np.linalg.norm(A, axis=0)
    X = A / norm
    B = np.abs(A @ A.T)
    
    np.fill_diagonal(B, 0.0)
    return np.max(B)

def test_distribuicao_do_maximo(n):
    """Testa a função distribuicao_do_maximo gerando matrizes Gaussianas
    e plota histogramas dos máximos calculados.

    Para cada execução, a função:
    1. Gera uma matriz Gaussiana com 100 linhas e 300 colunas.
    2. Calcula o máximo da distribuição dos produtos internos normalizados.
    3. Plota e salva o histograma do máximo calculado.

    Args:
        n (int): Número de execuções do teste.
    """
    maximos = np.empty(n, dtype=float)
    for i in range(n):
        A = generate_gaussian_matrix(100, 300)
        maximos[i] = distribuicao_do_maximo(A)

    hist, bin_edges = make_Histogram(maximos, bins=30)
    title = f"Distribuição do Máximo de Não-Ortogonalidade entre Colunas (K = {n})"
    # plot_histogram(hist, bin_edges, title=title, xlabel='Máximo', ylabel='Frequência', folder='figures/distribuicao_do_maximo')
    plot_histogram_seaborn(data = maximos, title=title, xlabel="Máximo",ylabel='Frequência', folder='figures/distribuicao_do_maximo' )

if __name__ == "__main__":
    
    print("Iniciando os testes...")
    
    # test_norma2_das_colunas()
    test_produto_interno()
    # time_start = time.time()
    # test_distribuicao_do_maximo(1000)
    

    # time_end = time.time()
    print("Teste concluído.")
    # print(f"Tempo total de execução: {time_end - time_start:.2f} segundos")
    # A = generate_gaussian_matrix(10000, 1000)
    # data = norma2_das_colunas(A)
    # melhor_ajuste = melhor_ajuste_distribuicao(data)
    # print(f"Melhor Ajuste (Distibuição na Matriz {10000}X{1000}: {melhor_ajuste} )")
    # # hist, bin_edges = make_Histogram(data, bins=30)
    # title = f"Normas 2 das Colunas - Matriz {10000}X{1000}"
    # # plot_histogram(hist, bin_edges, title=title, xlabel='Norma 2', ylabel='Frequência')
    # plot_histogram_seaborn(data, bins=25, title=title, xlabel='Norma 2',  ylabel='Frequência')
    