from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import difflib
import time

# Configurações
k = 10 # Número de folds no K-fold Validation
##

# Dados de treino
x = np.load("dados/x_treino.npy")
y = np.load("dados/y_treino.npy")
##

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
    model.add(Dense(units=73, activation='relu', kernel_initializer='random_uniform', input_dim = 73))
    model.add(Dropout(0.6))
    
    model.add(Dense(units = 128, activation = 'relu'))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = SGD(momentum=0.9, learning_rate=0.005)
    model.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
    return model


# Dataset teste
x_teste = np.load("dados/x_teste.npy")
y_teste = np.load("dados/y_teste.npy")
##

# Callbacks
es = EarlyStopping(monitor='val_loss', min_delta = 1e-10, patience = 20, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 3, verbose=0)
checkpoint = ModelCheckpoint(filepath='checkpoints/pesos.h5', monitor='loss',
                             save_best_only=True)
##


model = criarModelo()
kfold = StratifiedKFold(n_splits=k, shuffle=False)
cvscores = []

# K-fold
for treino, teste in kfold.split(x, y):
    history = model.fit(x[treino], y[treino], epochs=550, batch_size=20, verbose=0)
    scores = model.evaluate(x[teste], y[teste], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) # Resultado do Fold
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) # Resultado Final
##

# Treino Normal (Sem Kfold)
NOME = "Smartcare-Stress-Idoso-ANN-%s"%(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NOME))
h = model.fit(x,y, validation_split=0.2, epochs=500, batch_size=20)
##

# Gráfico Treino Normal
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
##

# Plot comparação dataset teste previsão
plotarComparacoes(model, x_teste, y_teste)
##