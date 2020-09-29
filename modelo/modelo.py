from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import difflib
import time
import matplotlib.pyplot as plt

# Configurações
n = 7 # Numero de horas anteriores para serem analisados
k = 10 # Número de folds no K-fold Validation

x = np.load("dados/x_treino.npy")
y = np.load("dados/y_treino.npy")


# Loop que vai formar os dados de treino multidimensionais para as redes LSTM
# com base no timestep n
def loopAnteriores(x, y):
    p = []
    s = []
    for i in range(n, len(x)):
        p.append(x[i-n:i, :])
        s.append(y[i])  
    p,s = np.array(p).astype("float32"), np.array(s).astype("int32")
    return p,s

# Função que plota um gráfico comparativo entre as classes reais e previstas
def plotarComparacoes(model, x_teste, y_teste, estatisticas=1):
    previsao = model.predict(x_teste)
    previsao = np.array([previsao > 0.5]).astype('int32')
    previsao = previsao.reshape(previsao.shape[1])
    
    if estatisticas:
        sm = difflib.SequenceMatcher(None,previsao, y_teste)
        print("\nEstatísticas")
        print("\tMédia das previsões: %.2f"%(previsao.mean()))
        print("\tMédia real: %.2f"%(y_teste.mean()))
        print("\tDesvio Padrão das previsões: %.2f"%(previsao.std()))
        print("\tDesvio Padrão: %.2f"%(y_teste.std()))
        print("\tPorcentagem de similaridade: %.2f %%"%(sm.ratio()*100))
    
    plt.plot(previsao, color = 'blue', label = 'Previsão', alpha=0.5)
    plt.plot(y_teste, color = 'red', label = 'Real', alpha=0.5)
    plt.title('Previsão de stress de idoso com base na movimentação')
    plt.xlabel('Intervalo de tempo')
    plt.ylabel('Stress')
    plt.legend()
    plt.show()


def criarModelo ():
    model = Sequential()
    model.add(Flatten(input_shape=(73, 2)))
    model.add(Dense(units = 128, activation='relu'))
    model.add(Dropout(0.5))
    
    for j in range(0, 2):
        model.add(Dense(units = 64, activation = 'relu'))
        model.add(Dropout(0.75))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
    return model


# Dataset teste
x_teste = np.load("dados/x_teste.npy")
y_teste = np.load("dados/y_teste.npy")

#x_teste, y_teste = loopAnteriores(x_teste, y_teste)

# Callbacks
es = EarlyStopping(monitor='loss', min_delta = 1e-10, patience = 20, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience = 3, verbose=0)
checkpoint = ModelCheckpoint(filepath='checkpoints/pesos.h5', monitor='loss',
                             save_best_only=True)


model = criarModelo()
kfold = StratifiedKFold(n_splits=k, shuffle=False)
cvscores = []

for treino, teste in kfold.split(x, y):
    history = model.fit(x[treino], y[treino], epochs=550, batch_size=128, verbose=0, callbacks=[es, rlr])
    scores = model.evaluate(x[teste], y[teste], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

NOME = "Smartcare-Stress-Idoso-ANN-%s"%(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NOME))
h = model.fit(x,y, validation_split=0.2, epochs=550, batch_size=16, callbacks=[tensorboard, es, rlr])
    
#print(results.mean())

#model.fit(x = x, y = y, batch_size = 128, epochs=500, 
#              callbacks=[es, rlr,tensorboard, checkpoint], validation_split=0.33,
#              verbose=0)

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('AcurÃ¡cia do modelo')
plt.ylabel('acurÃ¡cia')
plt.xlabel('Ã©poca')
plt.legend(['Treino', 'Teste'], loc='best')
plt.show()  
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Perda do modelo')
plt.ylabel('perda')
plt.xlabel('Ã©poca')
plt.legend(['train', 'test'], loc='best')
plt.show()

plotarComparacoes(model, x_teste, y_teste)
#"""