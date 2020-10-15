from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.utils import shuffle
import difflib
import time
from contextlib import redirect_stdout
from modulos.analise_dados import kfold_log_batches, plotarComparacoes, plotar, plotar_log, plotar_matriz_confusao
from modulos.machine_learning import carrega_dados, kfold

x, y, x_teste, y_teste = carrega_dados()

def criarModelo ():
    model = Sequential()
    model.add(Dense(units=73, activation='relu', kernel_initializer='random_uniform', input_dim = 73))
    model.add(Dropout(0.6))
    
    model.add(Dense(units = 128, activation = 'relu'))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = SGD(momentum=0.9, nesterov=True, learning_rate=0.0005)
    model.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
    return model




# Callbacks
es = EarlyStopping(monitor='val_loss', min_delta = 1e-10, patience = 20, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 3, verbose=0)
checkpoint = ModelCheckpoint(filepath='checkpoints/pesos.h5', monitor='loss',
                             save_best_only=True)
##
model = criarModelo()

for i in range(5):
    kfold(x, y, model)

batches = [50, 100]
h = kfold_log_batches(criarModelo, x, y, batches=batches)
plotar_log(h, batches)

# Treino Normal (Sem Kfold)
model = criarModelo()
NOME = "Smartcare-Stress-Idoso-ANN-%s"%(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/tensorboard/{}'.format(NOME))
h = model.fit(x,y, epochs=500, batch_size=100, verbose=0)
##

plotar(h)
plotar_matriz_confusao(model, x_teste, y_teste)

# Gráfico Treino Normal

##

# Plot comparação dataset teste previsão
plotarComparacoes(model, x_teste, y_teste)
##