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
    """
    Calcula a norma Euclidiana  de cada coluna de uma matriz.

    Esta função itera sobre as colunas da matriz de entrada e calcula a norma euclidiana
    de cada coluna individualmente, retornando uma lista com essas normas.

    Args:
        A (np.ndarray): A matriz de entrada (bidimensional).

    Returns:
        list: Uma lista contendo a norma euclidiana de cada coluna da matriz A.
    """
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



def plot_histogram_seaborn(data, bins=20, title='Histograma', xlabel='Valor', ylabel='Frequência', folder='figures/normas'):
    """
    Gera e salva um histograma com Estimativa de Densidade de Kernel (KDE) usando Seaborn e Matplotlib.

    Esta função cria um histograma para visualizar a distribuição dos dados fornecidos,
    sobrepõe uma curva KDE para estimar a função de densidade de probabilidade e
    adiciona uma linha vertical indicando a média dos dados. O gráfico resultante
    é salvo em um arquivo PNG no diretório especificado.

    Args:
        data (array-like): Os dados para os quais o histograma será plotado (e.g., lista, array NumPy).
        bins (int, optional): O número de bins (barras) do histograma. Padrão para 20.
        title (str, optional): O título principal do gráfico. A média e o desvio padrão
                                dos dados serão adicionados automaticamente ao título.
                                Padrão para 'Histograma'.
        xlabel (str, optional): O rótulo para o eixo X do gráfico. Padrão para 'Valor'.
        ylabel (str, optional): O rótulo para o eixo Y do gráfico. Padrão para 'Frequência'.
        folder (str, optional): O caminho do diretório onde o gráfico será salvo.
                                 O diretório será criado se não existir.
                                 Padrão para 'figures/normas'.

    Returns:
        None: A função salva o gráfico diretamente no sistema de arquivos.

    Notes:
        - A função calcula automaticamente a média e o desvio padrão dos `data` e os inclui
          no título e na legenda.
        - Utiliza `seaborn.histplot` para o histograma e KDE e `matplotlib.pyplot`
          para personalização e salvamento.
        - O nome do arquivo PNG é gerado a partir do `title`, convertendo espaços para
          underscores e letras para minúsculas.

    Raises:
        ImportError: Se 'os', 'numpy', 'matplotlib.pyplot' ou 'seaborn' não puderem ser importados.

    """

    os.makedirs(folder, exist_ok=True)

    filename = title.lower().replace(' ', '_')
    filepath = os.path.join(folder, f"{filename}.png")
    mean = np.mean(data)
    std = np.std(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True, stat='density',  color='skyblue', edgecolor='black', ax =ax)
    sns.kdeplot(data, color="purple", linewidth=2, label="KDE (Densidade)", ax =ax)
    ax.axvline(mean,        ls='--', lw=2,  color='blue',   label=f'Média: {mean:.2f}')
    ax.set_title(f"{title} (σ = {std:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Agora o Matplotlib encontra todos os rótulos
    ax.legend()

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo em: {filepath}")


def test_norma2_das_colunas():
    """
    Testa e visualiza a distribuição das normas euclidianas  das colunas de matrizes Gaussianas
    para diferentes dimensões.

    Esta função automatiza a criação de matrizes Gaussianas aleatórias de várias dimensões,
    calcula a norma euclidiana de cada uma de suas colunas e, em seguida, gera um histograma
    para visualizar a distribuição dessas normas. É útil para analisar como o
    comportamento das normas das colunas varia com as dimensões da matriz (linhas e colunas).

    A função testa combinações de linhas e colunas a partir dos seguintes valores:
    - Linhas (m): [1, 100, 1000, 10000]
    - Colunas (n): [1, 100, 1000, 10000]

    Para cada par (m, n):
    1. Gera uma matriz Gaussiana de dimensão m x n.
    2. Calcula a norma euclidiana de cada coluna da matriz.
    3. Plota um histograma da distribuição dessas normas, com um título descritivo
       e rótulos para os eixos.
    """
    valores_possiveis_linha =  [1, 100, 1000, 10000]
    valores_possiveis_coluna =  [1, 100, 1000, 10000]
    
    for linha in valores_possiveis_linha:
        for col in valores_possiveis_coluna:
    
            A = generate_gaussian_matrix(linha, col)
            data = norma2_das_colunas(A)
            title = f"Normas 2 das Colunas - Matriz {linha}X{col}"
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
        hist, bin_edges = make_Histogram(resultados, bins=30)
        title = f"Produto Interno das Colunas - Matriz {i}"
        plot_histogram_seaborn(data=resultados, bins =25,xlabel='Produto Interno', title=title, ylabel='Frequência', folder='figures/produto_interno' )
      
# =======================================
# ============= Question 3 ==============
# =======================================

def distribuicao_do_maximo(A, create_matrix=False, m = None, n = None):
    """
    Calcula o valor máximo da "não-ortogonalidade" entre pares de colunas de uma matriz.

    A "não-ortogonalidade" entre duas colunas é definida como o valor absoluto
    do produto interno normalizado entre elas, que é equivalente ao valor absoluto
    do cosseno do ângulo entre as colunas. Um valor próximo de 0 indica ortogonalidade
    e um valor próximo de 1 indica colinearidade.

    Esta função pode operar em uma matriz fornecida ou gerar uma nova matriz Gaussiana
    com dimensões especificadas. Ela normaliza as colunas da matriz para que tenham
    norma unitária, calcula a matriz de produtos internos (similar a uma matriz de correlação),
    zera a diagonal (para ignorar a auto-correlação das colunas consigo mesmas)
    e, finalmente, retorna o maior valor absoluto de não-ortogonalidade encontrado.

    Args:
        A (np.ndarray, optional): A matriz de entrada. Se `create_matrix` for True,
                                  este argumento será ignorado.
        create_matrix (bool, optional): Se True, uma nova matriz Gaussiana será gerada
                                        usando `generate_gaussian_matrix` com as dimensões `m` e `n`.
                                        Padrão para False.
        m (int, optional): Número de linhas da matriz a ser gerada, se `create_matrix` for True.
                           Deve ser fornecido se `create_matrix` for True.
        n (int, optional): Número de colunas da matriz a ser gerada, se `create_matrix` for True.
                           Deve ser fornecido se `create_matrix` for True.

    Returns:
        float: O valor máximo da não-ortogonalidade encontrado entre quaisquer pares
               de colunas distintas da matriz. Este valor estará no intervalo [0, 1].

    Raises:
        ValueError: Se `create_matrix` for True, mas `m` ou `n` não forem fornecidos.
        TypeError: Se `A` não for um `np.ndarray` e `create_matrix` for False.
    """
    if create_matrix:
        A = generate_gaussian_matrix(m, n) # gera  matriz caso ela ainda nn foi gerada
    norm = np.linalg.norm(A, axis=0)
    X = A / norm # normaliza as colunas
    B = np.abs(X.T @ X) # produto interno das colunas
    
    np.fill_diagonal(B, 0.0)
    return np.max(B)


def test_distribuicao_do_maximo(n, plot=True):
    """
    Testa a função `distribuicao_do_maximo` gerando múltiplas matrizes Gaussianas
    e analisando a distribuição dos valores máximos de não-ortogonalidade.

    Esta função executa a simulação `n` vezes. Em cada execução, ela:
    1. Gera uma nova matriz Gaussiana de 100 linhas por 300 colunas.
    2. Calcula o valor máximo da não-ortogonalidade entre quaisquer duas colunas
       distintas dessa matriz, usando `distribuicao_do_maximo`.
    3. Armazena esse valor máximo.

    Após todas as execuções, a função calcula e retorna a média, a moda e a
    mediana dos valores máximos observados. Opcionalmente, ela também gera e
    salva um histograma dessa distribuição de máximos para análise visual.

    Args:
        n (int): O número de vezes que a simulação será executada (ou seja, o número de
                 matrizes Gaussianas a serem geradas e analisadas).
        plot (bool, optional): Se `True`, um histograma da distribuição dos valores
                               máximos será gerado e salvo. Padrão para `True`.

    Returns:
        tuple: Uma tupla contendo três valores float:
               - `mean` (float): A média dos valores máximos de não-ortogonalidade calculados.
               - `moda` (float): A moda dos valores máximos de não-ortogonalidade calculados.
                                 Nota: Para dados contínuos, a moda pode não ser única ou
                                 ser sensível aos bins do histograma; aqui, ela retorna o
                                 valor mais frequente exato.
               - `mediana` (float): A mediana dos valores máximos de não-ortogonalidade calculados.
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
    """
    Calcula o estimador não enviesado da variância amostral (S²) de forma iterativa.

    Esta função calcula a soma dos quadrados dos desvios de cada ponto de dado em relação à
    média cumulativa dos dados até aquele ponto. No final, ela divide essa soma
    pelo número de dados menos 1 (`n-1`) para obter o estimador não enviesado da variância.

    Args:
        dados (array-like): Uma lista ou array NumPy de valores numéricos.

    Returns:
        float: O valor do estimador não enviesado da variância amostral (S²).

    Raises:
        ValueError: Se a entrada `dados` tiver menos de dois elementos, pois
                    a variância para um único ponto de dado não é definida
                    (divisão por zero ou por `len(dados) - 1`).
    """
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
    print(f"Valor do epsilon: {epsilon}")
    print(f"Valor do alpha: {alpha}")
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
    plt.plot(range(1, K_max + 1), medias, label='Média dos Máximos', color='#1f77b4', linewidth=2)
    # plt.plot([], [], ' ', label=f'Epsilon ($\epsilon$) = {epsilon:.4f}') # Adiciona uma entrada vazia na legenda


    plt.axhline(limite_superior, color='darkorange', linestyle='--', linewidth=2,
            label=f'Limite Superior: {limite_superior:.4f}')    
    plt.axhline(limite_inferior, color='chocolate', linestyle='-.', linewidth=2,
            label=f'Limite Inferior: {limite_inferior:.4f}')
    
    plt.axvline(K_teo, color='firebrick', linestyle='--', linewidth=2, label=f'K Teórico = {K_teo}')
    plt.axvline(K_real, color='darkgreen', linestyle='--', linewidth=2, label=f'K Real = {K_real}')
    # if K_real is not None:
    #     plt.axvspan(K_real, K_max, color='green', alpha=0.1, label='Região de Convergência Estável')

  
    plt.xlabel('K (Número de Simulações)', fontsize=14)
    plt.ylabel('Média dos Máximos', fontsize=14)
    plt.title(f'Estabilização da Média dos Máximos vs. K (${{\\epsilon}}$ = {epsilon:.4f})', fontsize=16, pad=20)
    
    plt.legend( framealpha=0.9, facecolor='white')

    plt.tight_layout()
    plt.savefig(f'figures/encontrar_K/media_maximos_vs_K_{epsilon:.4f}.png', dpi=300, bbox_inches='tight')
    plt.close()

    
    print(f"Análise com epsilon = {epsilon} finalizada")
    print()
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
    
    epsilon_poss =[0.005, 0.004, 0.003, 0.0015] 
    
    for eps in epsilon_poss:
        result = analise_K(m=100, n=300, epsilon=eps, alpha=0.05, K_max=3000)
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
    