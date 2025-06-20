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

def distribuicao_do_maximo(A, create_matrix=False, m = None, n = None):
    if create_matrix:
        A = generate_gaussian_matrix(m, n)
    norm = np.linalg.norm(A, axis=0)
    X = A / norm
    B = np.abs(X.T @ X)
    
    np.fill_diagonal(B, 0.0)
    return np.max(B)

def test_distribuicao_do_maximo(n, plot=True):
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

     #  Calculando Média
    mean = np.mean(maximos)
    # Calcula Moda 
    valores, contagens = np.unique(maximos, return_counts=True)
    moda = valores[np.argmax(contagens)]
    mediana = np.median(maximos)
    if plot:
        hist, bin_edges = make_Histogram(maximos, bins=30)
        title = f"Histograma do Máximo da Distribuição - Execução {n}"
        plot_histogram(hist, bin_edges, title=title, xlabel='Máximo', ylabel='Frequência', folder='figures/distribuicao_do_maximo')
    return mean, moda, mediana

# =======================================
# ============= Question 4 ==============
# =======================================

def compute_S2(dados):
    S2 = 0
    for i in range(len(dados)):
        Xbar = np.mean(dados[:i+1])
        S2 += (dados[i] - Xbar) ** 2
    S2 /= len(dados) - 1
    return S2


def encontrar_K_teorico(m, n, epsilon, alpha, K0=100, max_iter=1000):
    z = stats.norm.ppf(1 - alpha/2)
    K = K0
    dados = np.array(
        [distribuicao_do_maximo(A=None, create_matrix=True, m=m, n=n) for _ in range(K)]
    )

    for it in range(max_iter):
        S2 = compute_S2(dados)
        S = np.sqrt(S2)
        K_req = int(np.ceil((z * S / epsilon)**2))  

        if K_req <= K:
            ME = z * S / np.sqrt(K)
            if ME <= epsilon:
                # convergiu dentro da margem desejada
                K = K_req
                break
            # se ainda não convergiu, pedimos ao menos UMA réplica extra
            K_req = K + 1
        novas = np.array([
            distribuicao_do_maximo(A=None, create_matrix=True, m=m, n=n)
            for _ in range(K_req - K)
        ])
        dados = np.concatenate([dados, novas])
        K = K_req

    # calculamos estatísticas finais
    Xbar = dados.mean()
    S     = np.std(dados, ddof=1)
    ME    = z * S / np.sqrt(K)

    print(K, "é o valor de K encontrado com a tolerância de erro", ME)

    return {
        'K_final': K,
        'Xbar':    Xbar,
        'S':       S,
        'ME':      ME,
        'alpha':   alpha,
        'epsilon': epsilon,
        'dados':   dados
    }

def analise_K(m, n, epsilon, alpha, K_max):
    #Calcula K teórico e tolerância de erro    
    resultado = encontrar_K_teorico(m, n, epsilon, alpha)
    tol = resultado['ME']
    K_teo = resultado['K_final']
    S = resultado['S']
    if K_teo > K_max:
        raise ValueError(f"K teórico ({K_teo}) é maior que K máximo permitido ({K_max}). Ajuste K_max ou os parâmetros de entrada.")

    # Inicializa arrays para armazenar máximos e médias
    maximo = np.empty(K_max, dtype=float)
    medias = []
    # Gera matrizes e computa máximos
    for k in range(1, K_max + 1):
        A = generate_gaussian_matrix(m, n)
        maximo[k - 1] = distribuicao_do_maximo(A)
        medias.append(np.mean(maximo[:k]))

    # Calcula valor esperado e encontra K_real
    valor_esperado = np.mean(maximo[:K_teo])

    # Encontra todos os valores de K dentro da tolerância
    k_values = [k for k in range(1, K_max + 1) 
                if abs(medias[k - 1] - valor_esperado) < tol]

    print(k_values)
    # Determina K_real encontrando a região estável
    K_real = None
    for k in range(1, K_max + 1):
        if all(abs(medias[i] - valor_esperado) < tol 
               for i in range(k, K_max)):
            K_real = k
            break
            
    print(f"Valor de K encontrado: {K_real} (Teórico: {K_teo})")

    # Calcula limites de confiança
    limite_superior = valor_esperado + tol
    limite_inferior = valor_esperado - tol

    # Plota os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, K_max + 1), medias, label='Média dos Máximos')
    plt.axvline(K_teo, color='red', linestyle='--', label=f'K = {K_teo}')
    plt.axvline(K_real, color='green', linestyle='--', label=f'K Real = {K_real}')
    plt.axhline(limite_superior, color='orange', linestyle='--', 
                label=f'Limite Superior: {limite_superior:.4f}')
    plt.axhline(limite_inferior, color='orange', linestyle='--', 
                label=f'Limite Inferior: {limite_inferior:.4f}')
    plt.xlabel('K')
    plt.ylabel('Média dos Máximos')
    plt.title('Média dos Máximos vs. K')
    plt.legend()
    plt.savefig('figures/encontrar_K/media_maximos_vs_K.png')
    plt.close()

    return {
        'k_values': k_values,
        'valor_esperado': valor_esperado,
        'K_teo': K_teo,
        'K_real': K_real,
        'limite_superior': limite_superior,
        'limite_inferior': limite_inferior
    }


# =======================================
# ============= Question 5 ==============
# =======================================
    
    #  Calculando Média
    mean = np.mean(maximos)
    # Calcula Moda 
    valores, contagens = np.unique(maximos, return_counts=True)
    moda = valores[np.argmax(contagens)]
    mediana = np.median(maximos)

    hist, bin_edges = make_Histogram(maximos, bins=30)
    title = f"Distribuição do Máximo de não ortogonalidade entre colunas"
    plot_histogram_seaborn(data=maximos, title= title,  bins=25,  xlabel='Máximo', ylabel='Frequência', folder='figures/distribuicao_do_maximo') 
    
    
    return mean, moda, mediana

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


if __name__ == "__main__":
    time_start = time.time()
    print("Iniciando os testes...")
    
    # test_norma2_das_colunas()
    # test_produto_interno()
    
    # test_distribuicao_do_maximo(1000)
    # k_values = np.array(encontrar_K(1000))
    # k_value = test_encontrar_K(1000)
    # print(f"Valor de K encontrado: {k_value}")
    # result = encontrar_K_teorico(100, 300, epsilon=0.01, alpha=0.05)
    # print(encontrar_K(0.001, 0.05, K_max=1000))
    result = analise_K(m=100, n=300, epsilon=0.001, alpha=0.05, K_max=3000)
    print(f"Valor esperado: {result['valor_esperado']:.4f}, K Teórico: {result['K_teo']}, K Real: {result['K_real']}")
    
    # print(len(k_values), " valores de K encontrados.")
    # test_distribuicao_do_maximo_parte_2(1000)
    time_end = time.time()
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
    