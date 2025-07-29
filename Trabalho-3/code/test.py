import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Cria os diretórios para salvar as imagens e tabelas, se ainda não existirem
os.makedirs("./code/images", exist_ok=True)
os.makedirs("./code/tables", exist_ok=True)

# Parâmetros
velocidadeAdveccao = 1.0        # Velocidade de advecção
numeroCourant = 0.8             # Número de Courant
limiteXMinimo, limiteXMaximo = 0.0, 1.0
tempoFinal1, tempoFinal3, tempoFinal5 = 1.0, 3.0, 5.0
numPontosEspaco = 200           # Número de pontos no espaço
intervaloEspacial = (limiteXMaximo - limiteXMinimo) / numPontosEspaco
intervaloTempo = numeroCourant * intervaloEspacial / velocidadeAdveccao  # Intervalo de tempo para satisfazer CFL

# Condição inicial
posicaoEspacial = np.linspace(limiteXMinimo, limiteXMaximo, numPontosEspaco, endpoint=False)
condicaoInicial = 1.5 * np.exp(-200 * (posicaoEspacial - 0.3)**2) + \
                  np.where((posicaoEspacial >= 0.6) & (posicaoEspacial <= 0.8), 1.5, 0.0)

# Limitadores
def limitadorOsher(theta):
    return np.maximum(0, np.minimum(1, theta))

def limitadorSweby(theta, beta=1.5):
    return np.maximum(0, np.minimum(beta * theta, np.minimum(1, theta)))

def limitadorVanAlbada(theta):
    return (theta + theta**2) / (1 + theta**2 + 1e-6)

# Método TVD corrigido
def metodoTvd(densidade, intervaloTempo, intervaloEspacial, numeroCourant, limitador):
    c = numeroCourant
    epsilon = 1e-6
    u = velocidadeAdveccao
    numPontosEspaco = len(densidade)
    novaDensidade = densidade.copy()
    for i in range(numPontosEspaco):
        # Índices com condições de contorno periódicas
        esquerda2 = (i - 2) % numPontosEspaco
        esquerda1 = (i - 1) % numPontosEspaco
        direita1 = (i + 1) % numPontosEspaco
        
        # Cálculo dos deltas
        deltaPhiI = densidade[i] - densidade[esquerda1]
        deltaPhiDireita1 = densidade[direita1] - densidade[i]
        deltaPhiEsquerda1 = densidade[esquerda1] - densidade[esquerda2]
        
        # Cálculo de theta
        denomThetaI = deltaPhiDireita1 + epsilon
        denomThetaEsquerda1 = deltaPhiI + epsilon
        thetaI = deltaPhiI / denomThetaI
        thetaEsquerda1 = deltaPhiEsquerda1 / denomThetaEsquerda1
        
        # Aplicação do limitador
        phiLimI = limitador(thetaI)
        phiLimEsquerda1 = limitador(thetaEsquerda1)
        
        # Fluxos numéricos
        fluxoDireitaMeio = u * densidade[i] + u * (1 - c) / 2 * phiLimI * deltaPhiDireita1
        fluxoEsquerdaMeio = u * densidade[esquerda1] + u * (1 - c) / 2 * phiLimEsquerda1 * deltaPhiI
        
        # Atualização da densidade
        novaDensidade[i] = densidade[i] - c * (fluxoDireitaMeio - fluxoEsquerdaMeio)
    return novaDensidade

# Função genérica para resolver usando diferentes limitadores
def resolverAdveccaoTvd(metodoLimitador, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal):
    densidade = condicaoInicial.copy()
    tempoAtual = 0
    while tempoAtual < tempoFinal:
        densidade = metodoTvd(densidade, intervaloTempo, intervaloEspacial, numeroCourant, metodoLimitador)
        tempoAtual += intervaloTempo
    return densidade

# Função para verificar a condição de estabilidade CFL
def verificarEstabilidadeCFLPorMetodo(metodoLimitador, condicaoInicial, intervaloEspacialInicial, numeroCourant, tempoFinal, nomeMetodo):
    resultadosEstabilidade = {}
    fatoresReducao = [1, 2, 4, 8]  # Fatores para reduzir dx e dt
    for fator in fatoresReducao:
        # Ajusta dx e dt
        dx = intervaloEspacialInicial / fator
        dt = numeroCourant * dx / velocidadeAdveccao
        posicoes = np.linspace(limiteXMinimo, limiteXMaximo, numPontosEspaco * fator, endpoint=False)
        densidadeInicial = 1.5 * np.exp(-200 * (posicoes - 0.3)**2) + \
                           np.where((posicoes >= 0.6) & (posicoes <= 0.8), 1.5, 0.0)
        
        # Resolução numérica
        densidadeFinal = resolverAdveccaoTvd(metodoLimitador, densidadeInicial, dt, dx, numeroCourant, tempoFinal)
        
        # Armazena os resultados
        resultadosEstabilidade[f"dx={dx:.4f}, dt={dt:.4f}"] = {
            "posicoes": posicoes,
            "densidadeFinal": densidadeFinal,
            "densidadeInicial": densidadeInicial
        }
    return resultadosEstabilidade

# Execução dos métodos para os tempos t=1, t=3 e t=5
metodos = {
    "Osher": limitadorOsher,
    "Sweby": limitadorSweby,
    "Van Albada": limitadorVanAlbada
}

# Filtrando para incluir apenas passos de tempo múltiplos de 0.05
passosTempo = np.arange(0, len(posicaoEspacial), int(0.05 / intervaloEspacial))  # Seleciona com base no intervalo

resultados = {}
for nomeMetodo, limitador in metodos.items():
    plt.figure(figsize=(12, 10))
    
    # Soluções para t=1, t=3, t=5
    resultados[nomeMetodo] = {
        "t=1": resolverAdveccaoTvd(limitador, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal1),
        "t=3": resolverAdveccaoTvd(limitador, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal3),
        "t=5": resolverAdveccaoTvd(limitador, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal5)
    }

    # Verificação da condição de estabilidade CFL
    resultadosEstabilidade = verificarEstabilidadeCFLPorMetodo(
        limitador, condicaoInicial, intervaloEspacial, numeroCourant, tempoFinal1, nomeMetodo
    )
    
    # Criando tabelas LaTeX com a coluna de posição para estabilidade
    tabela_dados = pd.DataFrame({
        "Posicao Espacial": posicaoEspacial[passosTempo],
        "Condicao Inicial": condicaoInicial[passosTempo],
        f"{nomeMetodo} t=1": resultados[nomeMetodo]["t=1"][passosTempo],
        f"{nomeMetodo} t=3": resultados[nomeMetodo]["t=3"][passosTempo],
        f"{nomeMetodo} t=5": resultados[nomeMetodo]["t=5"][passosTempo]
    })

    # Adicionando colunas de estabilidade
    for chave, dados in resultadosEstabilidade.items():
        tabela_dados[f"Estab {chave}"] = dados["densidadeFinal"][passosTempo]

    # Salvando a tabela como LaTeX
    tabela_dados.to_latex(f"./code/tables/{nomeMetodo}_completo.tex", index=False)

    # Salvando o gráfico
    plt.tight_layout()
    plt.savefig(f"./code/images/{nomeMetodo}.png")  # Salvamento da imagem com os quatro gráficos
    plt.close()