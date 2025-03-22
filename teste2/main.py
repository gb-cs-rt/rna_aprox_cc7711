#pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
import os

print('Carregando Arquivo de teste')
arquivo = np.load('teste2/teste2.npy')
x = arquivo[0]

scale = MaxAbsScaler().fit(arquivo[1])
y = np.ravel(scale.transform(arquivo[1]))

num_iteracoes = 10

# Definir diferentes arquiteturas de rede neural
arquiteturas = [
    (15,),  # Uma camada com 15 neurônios
    (15, 8),  # Duas camadas com 10 neurônios cada
    (15, 10, 5)  # Três camadas com 20, 10 e 5 neurônios
]

# Definir número de iterações para cada arquitetura
iteracoes_por_arquitetura = {
    (15,): 1200,
    (15, 8): 1000,
    (15, 10, 5): 600
}

# Definir pastas para salvar os resultados
pastas = ['simulacao1', 'simulacao2', 'simulacao3']

for idx, arquitetura in enumerate(arquiteturas):
    print(f'\nSimulação com arquitetura: {arquitetura}')
    erros_finais = []
    pasta = pastas[idx]
    caminho_pasta = f'teste2/images/{pasta}'

    # Criar pasta se não existir
    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)

    iteracoes = iteracoes_por_arquitetura[arquitetura]
    i = 0
    while i < num_iteracoes:
        regr = MLPRegressor(hidden_layer_sizes=arquitetura,
                            max_iter=iteracoes,
                            activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                            solver='adam', #{‘lbfgs’, ‘sgd’, ‘adam’}
                            learning_rate='adaptive',
                            n_iter_no_change=iteracoes,
                            verbose=False)
        print(f'Treinando RNA - Iteração {i+1}')
        regr = regr.fit(x, y)

        print('Preditor')
        y_est = regr.predict(x)

        plt.figure(figsize=[14, 7])

        # plot curso original
        plt.subplot(1, 3, 1)
        plt.title('Função Original')
        plt.plot(x, y, color='green')

        # plot aprendizagem
        plt.subplot(1, 3, 2)
        plt.title('Curva erro (%s)' % str(round(regr.best_loss_, 5)))
        plt.plot(regr.loss_curve_, color='red')
        print(regr.best_loss_)

        # plot regressor
        plt.subplot(1, 3, 3)
        plt.title('Função Original x Função aproximada')
        plt.plot(x, y, linewidth=1, color='green')
        plt.plot(x, y_est, linewidth=2, color='blue')

        plt.savefig(f'{caminho_pasta}/iteracao_{i+1}_arquitetura_{arquitetura}.png')
        plt.close()

        erros_finais.append(regr.best_loss_)
        i += 1

    # Calcular média e desvio padrão dos erros finais
    media_erro = np.mean(erros_finais)
    desvio_padrao_erro = np.std(erros_finais)

    print(f'Média do erro final para arquitetura {arquitetura}: {media_erro}')
    print(f'Desvio padrão do erro final para arquitetura {arquitetura}: {desvio_padrao_erro}')

    # Salvar média e desvio padrão em um arquivo de texto
    with open(f'{caminho_pasta}/resultados.txt', 'w') as f:
        f.write(f'Média do erro final para arquitetura {arquitetura}: {media_erro}\n')
        f.write(f'Desvio padrão do erro final para arquitetura {arquitetura}: {desvio_padrao_erro}\n')




