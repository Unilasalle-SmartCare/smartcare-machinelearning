from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_pickle("dados/dataset-treino.df")
df_treinamento = df
df = np.array(df)

previsores = []
stress = []

n = 79 # Numero de registros anteriores para serem analisados

for i in range(n, len(df)):
    previsores.append(df[i-n:i, 1:3])
    stress.append(df[i,3])
previsores, stress = np.array(previsores).astype('float64'), np.array(stress)

def criarModelo (optimizer, loss, kernel_initializer, neurons, dropout):
    model = Sequential()
    model.add(LSTM(units = 150, return_sequences = True, input_shape = (79, 2)))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units = neurons, return_sequences = True))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units = neurons, return_sequences = True))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units = neurons))
    model.add(Dropout(dropout))
    
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    model.compile(optimizer = optimizer, loss = loss,
                      metrics = ['mean_absolute_error'])
    return model

es = EarlyStopping(monitor='loss', min_delta = 1e-10, patience = 10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience = 5, verbose=1)
checkpoint = ModelCheckpoint(filepath='checkpoints/pesos.h5', monitor='loss',
                             save_best_only=True)
model = KerasRegressor(build_fn=criarModelo)
parametros = {
    'batch_size': [32, 64],
    'epochs': [100],
    'optimizer': ['adam', 'sgd'],
    'loss': ['mean_absolute_error', 'binary_crossentropy'],
    'neurons': [50, 100],
    'dropout': [0.3, 0.5]
}

grid_search = GridSearchCV(estimator=model, param_grid=parametros, scoring='neg_mean_absolute_error', cv=10)
grid_search = grid_search.fit(previsores, stress, callbacks=[es,rlr, checkpoint])
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


df_teste = pd.read_pickle("dados/dataset-teste.df")
stress_teste = df_teste.iloc[:, 3].values
frames = [df_treinamento, df_teste]

base_completa = pd.concat(frames)
base_completa = base_completa.drop('date', axis = 1)
base_completa = base_completa.drop('stress', axis = 1)


entradas = base_completa[len(base_completa) - len(df_teste) - n:].values

normalizador = joblib.load("dados/normalizador.save")

X_teste = []
for i in range(n, len(entradas)):
    X_teste.append(entradas[i-n:i, :])
X_teste = np.array(X_teste)

normalizador_previsao = joblib.load("dados/normalizador-previsao.save")

previsao = model.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsao)

previsoes = np.array(previsoes).astype(int)

previsoes.mean()
stress_teste.mean()

plt.plot(stress_teste, color = 'red', label = 'Idoso Estressado?')
plt.plot(previsoes, color = 'blue', label = 'Previsão')
plt.title('Previsão de stress de idoso com base na movimentação')
plt.xlabel('Tempo')
plt.ylabel('Stress')
plt.legend()
plt.show()
