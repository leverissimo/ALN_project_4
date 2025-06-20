import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from src.utils import *
import warnings
from scipy.optimize import OptimizeWarning


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1 - Testar norma 2 das colunas")
        print("2 - Testar produto interno das colunas")
        print("3 - Testar distribuição do máximo da não-ortogonalidade")
        print("4 - Análise para encontrar K teórico e real")
        print("5 - Testar distribuição do máximo - parte 2")
        print("0 - Sair")

        try:
            escolha = int(input("Escolha uma opção: "))
        except ValueError:
            print("Por favor, insira um número válido.")
            continue

        if escolha == 0:
            print("Saindo...")
            break
        elif escolha == 1:
            print("Executando teste da norma 2 das colunas...")
            test_norma2_das_colunas()
        elif escolha == 2:
            print("Executando teste do produto interno...")
            test_produto_interno()
        elif escolha == 3:
            n = input("Quantas simulações deseja executar? (exemplo: 1000) ")
            try:
                n = int(n)
            except:
                print("Número inválido. Usando 1000.")
                n = 1000
            mean, moda, mediana = test_distribuicao_do_maximo(n)
            print(f"Média: {mean:.4f}, Moda: {moda:.4f}, Mediana: {mediana:.4f}")
        elif escolha == 4:
            print("Análise para encontrar K teórico e real.")
            try:
                m = int(input("Digite o valor de m (linhas da matriz): "))
                n = int(input("Digite o valor de n (colunas da matriz): "))
                epsilon = float(input("Digite epsilon (exemplo: 0.005): "))
                alpha = float(input("Digite alpha (exemplo: 0.05): "))
                K_max = int(input("Digite o valor máximo para K (exemplo: 3000): "))
            except Exception as e:
                print(f"Entrada inválida: {e}. Voltando ao menu.")
                continue
            try:
                resultado = analise_K(m, n, epsilon, alpha, K_max)
                print(f"Valor esperado: {resultado['valor_esperado']:.4f}")
                print(f"K Teórico: {resultado['K_teo']}")
                print(f"K Real: {resultado['K_real']}")
            except Exception as e:
                print(f"Erro durante análise: {e}")
        elif escolha == 5:
            n = input("Quantas simulações deseja executar para parte 2? (exemplo: 1000) ")
            try:
                n = int(n)
            except:
                print("Número inválido. Usando 1000.")
                n = 1000
            test_distribuicao_do_maximo_parte_2(n)
            print("Teste parte 2 concluído. Verifique as figuras geradas.")
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
