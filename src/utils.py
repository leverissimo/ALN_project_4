import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

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

def melhor_ajuste_distribuicao(data):
    # Distributions to test - focusing on likely candidates
    distribuicoes = [
        ('norm', stats.norm),
        ('chi', stats.chi),
        ('chi2', stats.chi2),
        ('rayleigh', stats.rayleigh)
    ]
    
    melhor_p = -1
    melhor_nome = ''
    
    for nome, distrib in distribuicoes:
        try:
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    plt.figure(figsize=(10, 6))
    # sns.set(style="darkgrid")  # ou "whitegrid", "ticks"...
    sns.histplot(data, bins=bins, kde=True, stat='density',  color='skyblue', edgecolor='black', ax =ax)
    sns.kdeplot(data, color="purple", linewidth=2, label="KDE (Densidade)", ax =ax)
    ax.axvline(mean,        ls='--', lw=2,  color='blue',   label=f'Média: {mean:.2f}')
    # sns.kdeplot(data, color="red
    # ", linewidth=2)
    # Títulos e eixos
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Agora o Matplotlib encontra todos os rótulos
    ax.legend()

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo em: {filepath}")





# def plot_histogram_seaborn(data, *, bins=20, title='Histograma',
#                            xlabel='Valor', ylabel='Frequência',
#                            folder='figures/normas'):
#     os.makedirs(folder, exist_ok=True)

#     filename = title.lower().replace(' ', '_')
#     filepath = os.path.join(folder, f"{filename}.png")

#     # --- Crie UMA figura e UM eixo ----------------------------
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.set_style("darkgrid")

#     # Histograma (stat='density' p/ ficar na mesma escala da KDE)
#     sns.histplot(
#         data,
#         bins=bins,
#         stat='density',
#         color='skyblue',
#         edgecolor='black',
#         label='Histograma',
#         ax=ax                       # <- mesmo eixo
#     )

#     # KDE sobre o mesmo eixo, com rótulo
#     sns.kdeplot(
#         data,
#         color='purple',
#         linewidth=2,
#         label='KDE (Densidade)',
#         ax=ax                       # <- mesmo eixo
#     )

#     # Linhas de média ± desvio‑padrão (opcional)
#     mean = np.mean(data)
#     std  = np.std(data)
#     ax.axvline(mean,        ls='--', lw=2,  color='blue',   label=f'Média: {mean:.2f}')
#     ax.axvline(mean - std,  ls='--', lw=1.5,color='green',  label=f'-1σ: {mean - std:.2f}')
#     ax.axvline(mean + std,  ls='--', lw=1.5,color='green',  label=f'+1σ: {mean + std:.2f}')
#     #  • Se quiser sombrear a área ±1σ:
#     # ax.axvspan(mean - std, mean + std, color='green', alpha=0.2, label='±1σ')

#     # Títulos e eixos
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     # Agora o Matplotlib encontra todos os rótulos
#     ax.legend()

#     fig.savefig(filepath, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Gráfico salvo em: {filepath}")

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

    intervalos = [100, 200, 500, 1000]
    for i in intervalos:
        A = generate_gaussian_matrix(100, i)
        resultados = produto_interno(A)
        hist, bin_edges = make_Histogram(resultados, bins=30)
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
<<<<<<< HEAD
    title = f"Histograma do Máximo da Distribuição - Execução {n}"
    plot_histogram(hist, bin_edges, title=title, xlabel='Máximo', ylabel='Frequência', folder='figures/distribuicao_do_maximo')


def test_distribuicao_do_maximo_parte_2(n):
    maximos = np.empty(n, dtype=float)
    pares_mn = [(100, 100), (100, 300), (200, 200), (200, 600), (500, 500), (500, 1500), (1000, 1000), (1000, 3000)]
    for p in pares_mn:
        for i in range(n):
            A = generate_gaussian_matrix(p[0], p[1])
            maximos[i] = distribuicao_do_maximo(A)

        hist, bin_edges = make_Histogram(maximos, bins=30)
        title = f"Histograma do Máximo da Distribuição - Execução {n}, Dimensões {p[0]}X{p[1]}"
        plot_histogram(hist, bin_edges, title=title, xlabel='Máximo', ylabel='Frequência', folder=f'figures/distribuicao_do_maximo/parte_2/valores_de_K/{n}')
        maximos = np.empty(n, dtype=float)

=======
    title = f"Distribuição do Máximo de Não-Ortogonalidade entre Colunas (K = {n})"
    # plot_histogram(hist, bin_edges, title=title, xlabel='Máximo', ylabel='Frequência', folder='figures/distribuicao_do_maximo')
    plot_histogram_seaborn(data = maximos, title=title, xlabel="Máximo",ylabel='Frequência', folder='figures/distribuicao_do_maximo' )
>>>>>>> c212ec271de10b7c371678c9971e9293eb79fbe9

if __name__ == "__main__":
    
    print("Iniciando os testes...")
    
<<<<<<< HEAD
    # test_norma2_das_colunas(5)
    # test_produto_interno()
    
    # test_distribuicao_do_maximo(1000)
    test_distribuicao_do_maximo_parte_2(1000)
=======
    test_norma2_das_colunas()
    test_produto_interno()
    time_start = time.time()
    test_distribuicao_do_maximo(1000)
>>>>>>> c212ec271de10b7c371678c9971e9293eb79fbe9

    time_end = time.time()
    print("Teste concluído.")
    print(f"Tempo total de execução: {time_end - time_start:.2f} segundos")