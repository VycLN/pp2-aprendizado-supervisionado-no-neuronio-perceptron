import numpy as np
import matplotlib as plt

# Função degrau de ativação do neurônio
def funcao_ativacao(soma, teta):
    if (soma >= teta):
        return 1
    return 0

# Função que realiza o produto escalar de um regitro com os pesos
def calculo_saida(registro, pesos, teta):
    s = registro.dot(pesos)
    return funcao_ativacao(s, teta)

# Função de treinamento que é executada até que não haja erro
def treinamento(entradas, saidas, pesos, taxa_aprendizagem, teta):
    # Inicialização de variáveis
    erro_total = None
    epocas = 0
    ajustes_por_epoca = []
    # Loop que garante a execução até que não haja erro
    while (erro_total != 0):
        erro_total = 0
        ajustes_pesos = 0
        # Loop para percorrer todas as entradas/saídas
        for i in range(0, len(entradas), 1):
            alterou_pesos = False
            # Cálculo da saída para a entrada atual aplicando-se os pesos
            saida_calculada = calculo_saida(np.asarray(entradas[i]), pesos, teta)
            # Cálculo do erro da saída
            erro = saidas[i] - saida_calculada
            # Acumulação do erro total
            erro_total += erro
            # Loop para atualização do vetor de pesos
            for j in range(0, len(pesos), 1):
                # Cálculo do novo peso
                aux = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
                # Contagem de ajustes dos pesos da época caso haja mudança de valor
                if(aux != pesos[j]):
                    ajustes_pesos += 1
                    alterou_pesos = True
                pesos[j] = aux
            # Exibe os pesos casa haja alteração
            if(alterou_pesos == True):
                print("Pesos: ", pesos)
        # Exibe o número de ajustes de pesos da época
        print("Número de ajustes no vetor de pesos: ", ajustes_pesos)
        # Guarda o total de ajustes por época
        ajustes_por_epoca.append(ajustes_pesos)
        epocas += 1
    return pesos, epocas, ajustes_por_epoca

'''
# Caso de teste para operador E (lógico)
entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
saidas = np.array([0,0,0,1])

pesos = np.array([0.0, 0.0])

taxa_aprendizagem = 0.1
teta = 1

pesos, epocas, ajustes = treinamento(entradas, saidas, pesos, taxa_aprendizagem, teta)

print("Épocas: ", epocas)
print("Pesos: ", pesos)
print("Ajustes", sum(ajustes))
print(calculo_saida(entradas[0], pesos, teta))
print(calculo_saida(entradas[1], pesos, teta))
print(calculo_saida(entradas[2], pesos, teta))
print(calculo_saida(entradas[3], pesos, teta))
'''