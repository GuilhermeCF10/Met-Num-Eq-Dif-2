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
numPontosEspaco = 100           # Número de pontos no espaço
intervaloEspacial = (limiteXMaximo - limiteXMinimo) / numPontosEspaco
intervaloTempo = numeroCourant * intervaloEspacial / velocidadeAdveccao  # Intervalo de tempo para satisfazer CFL

# Condição inicial
posicaoEspacial = np.linspace(limiteXMinimo, limiteXMaximo, numPontosEspaco, endpoint=False)
condicaoInicial = 1.5 * np.exp(-200 * (posicaoEspacial - 0.3)**2) + \
                  np.where((posicaoEspacial >= 0.6) & (posicaoEspacial <= 0.8), 1.5, 0.0)

# Método Upwind com condições de contorno periódicas
def metodoUpwind(densidade, intervaloTempo, intervaloEspacial, numeroCourant):
    """
    Calcula a solução de advecção usando o método Upwind.
    """
    novaDensidade = densidade.copy()
    for i in range(numPontosEspaco):
        novaDensidade[i] = densidade[i] - numeroCourant * (densidade[i] - densidade[i-1])
    return novaDensidade

# Método Lax-Wendroff com condições de contorno periódicas
def metodoLaxWendroff(densidade, intervaloTempo, intervaloEspacial, numeroCourant):
    """
    Calcula a solução de advecção usando o método Lax-Wendroff.
    """
    novaDensidade = densidade.copy()
    for i in range(numPontosEspaco):
        novaDensidade[i] = densidade[i] - 0.5 * numeroCourant * (densidade[(i+1) % numPontosEspaco] - densidade[i-1]) + \
                           0.5 * numeroCourant**2 * (densidade[i-1] - 2 * densidade[i] + densidade[(i+1) % numPontosEspaco])
    return novaDensidade

# Método Beam-Warming com condições de contorno periódicas
def metodoBeamWarming(densidade, intervaloTempo, intervaloEspacial, numeroCourant):
    """
    Calcula a solução de advecção usando o método Beam-Warming.
    """
    novaDensidade = densidade.copy()
    for i in range(numPontosEspaco):
        novaDensidade[i] = densidade[i] - numeroCourant * (densidade[i] - densidade[i-1]) - \
                           0.5 * numeroCourant * (1 - numeroCourant) * (densidade[i] - 2 * densidade[i-1] + densidade[i-2])
    return novaDensidade

# Método Fromm com condições de contorno periódicas
def metodoFromm(densidade, intervaloTempo, intervaloEspacial, numeroCourant):
    """
    Calcula a solução de advecção usando o método Fromm.
    """
    novaDensidade = densidade.copy()
    for i in range(numPontosEspaco):
        novaDensidade[i] = densidade[i] - 0.25 * numeroCourant * (densidade[(i+1) % numPontosEspaco] + 3 * densidade[i] - \
                             5 * densidade[i-1] + densidade[i-2]) + \
                           0.25 * numeroCourant**2 * (densidade[(i+1) % numPontosEspaco] - densidade[i] - \
                                                      densidade[i-1] + densidade[i-2])
    return novaDensidade

# Função para resolver a advecção com diferentes métodos numéricos
def resolverAdveccao(metodo, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal):
    """
    Calcula a solução da advecção para um determinado método e tempo final.
    """
    densidade = condicaoInicial.copy()
    tempoAtual = 0
    while tempoAtual < tempoFinal:
        densidade = metodo(densidade, intervaloTempo, intervaloEspacial, numeroCourant)
        tempoAtual += intervaloTempo
    return densidade

# Execução dos métodos para os tempos t=1, t=3 e t=5
densidadeUpwind1 = resolverAdveccao(metodoUpwind, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal1)
densidadeLax1 = resolverAdveccao(metodoLaxWendroff, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal1)
densidadeBeam1 = resolverAdveccao(metodoBeamWarming, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal1)
densidadeFromm1 = resolverAdveccao(metodoFromm, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal1)

densidadeUpwind3 = resolverAdveccao(metodoUpwind, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal3)
densidadeLax3 = resolverAdveccao(metodoLaxWendroff, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal3)
densidadeBeam3 = resolverAdveccao(metodoBeamWarming, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal3)
densidadeFromm3 = resolverAdveccao(metodoFromm, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal3)

densidadeUpwind5 = resolverAdveccao(metodoUpwind, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal5)
densidadeLax5 = resolverAdveccao(metodoLaxWendroff, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal5)
densidadeBeam5 = resolverAdveccao(metodoBeamWarming, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal5)
densidadeFromm5 = resolverAdveccao(metodoFromm, condicaoInicial, intervaloTempo, intervaloEspacial, numeroCourant, tempoFinal5)

# Plotagem dos resultados
metodos = {
    "Upwind": (densidadeUpwind1, densidadeUpwind3, densidadeUpwind5),
    "Lax-Wendroff": (densidadeLax1, densidadeLax3, densidadeLax5),
    "Beam-Warming": (densidadeBeam1, densidadeBeam3, densidadeBeam5),
    "Fromm": (densidadeFromm1, densidadeFromm3, densidadeFromm5)
}

for nomeMetodo, (resultadoT1, resultadoT3, resultadoT5) in metodos.items():
    plt.figure(figsize=(15, 5))
    
    # Gráfico para t=1
    plt.subplot(1, 3, 1)
    plt.plot(posicaoEspacial, condicaoInicial, label='Cond. Inicial', linestyle='--')
    plt.plot(posicaoEspacial, resultadoT1, label=f'{nomeMetodo} t=1')
    plt.title(f'Solução {nomeMetodo} para t=1')
    plt.legend()
    
    # Gráfico para t=3
    plt.subplot(1, 3, 2)
    plt.plot(posicaoEspacial, condicaoInicial, label='Cond. Inicial', linestyle='--')
    plt.plot(posicaoEspacial, resultadoT3, label=f'{nomeMetodo} t=3')
    plt.title(f'Solução {nomeMetodo} para t=3')
    plt.legend()

    # Gráfico para t=5
    plt.subplot(1, 3, 3)
    plt.plot(posicaoEspacial, condicaoInicial, label='Cond. Inicial', linestyle='--')
    plt.plot(posicaoEspacial, resultadoT5, label=f'{nomeMetodo} t=5')
    plt.title(f'Solução {nomeMetodo} para t=5')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"./code/images/{nomeMetodo}.png")  # Salvamento da imagem
    # plt.show()

    # Filtrando para incluir mais pontos, pegando a cada quinto índice
    filtro = np.arange(0, len(posicaoEspacial), 5)  # Seleciona a cada 5 pontos
    dados = pd.DataFrame({
        "Posição Espacial": posicaoEspacial[filtro],
        "Cond. Inicial": condicaoInicial[filtro],
        f"{nomeMetodo} t=1": densidadeUpwind1[filtro],
        f"{nomeMetodo} t=3": densidadeUpwind3[filtro],
        f"{nomeMetodo} t=5": densidadeUpwind5[filtro]
    }).to_latex(f"./code/tables/{nomeMetodo}.tex")



