#pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

print('Carregando Arquivo de teste')
arquivo = np.load('teste1.npy')
x = arquivo[0]


scale= MaxAbsScaler().fit(arquivo[1])
y = np.ravel(scale.transform(arquivo[1]))


iteracoes = 400

regr = MLPRegressor(hidden_layer_sizes=(15,5),
                    max_iter=iteracoes,
                    activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam', #{‘lbfgs’, ‘sgd’, ‘adam’}
                    #loss_curve_ = 5, #se lbfgs
                    learning_rate = 'adaptive',
                    n_iter_no_change=iteracoes,
                    verbose=False)
print('Treinando RNA')
regr = regr.fit(x,y)



print('Preditor')
y_est = regr.predict(x)






plt.figure(figsize=[14,7])

#plot curso original

plt.subplot(1,3,1)
plt.title('Função Original')
plt.plot(x,y,color='green')


#plot aprendizagem

plt.subplot(1,3,2)
plt.title('Curva erro (%s)' % str(round(regr.best_loss_,5)))
plt.plot(regr.loss_curve_,color='red')
print(regr.best_loss_)

#plot regressor
plt.subplot(1,3,3)
plt.title('Função Original x Função aproximada')
plt.plot(x,y,linewidth=1,color='green')
plt.plot(x,y_est,linewidth=2,color='blue')
plt.show()




