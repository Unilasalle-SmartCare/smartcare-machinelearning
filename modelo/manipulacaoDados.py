import pandas as pd
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Função que pega o shape do horário com mais pontos
def maiorShapeLista(array):
    maior = [0,0]
    for a in array:
       if a.shape[0] > maior[0]:
           maior[0] = a.shape[0]
           
       if a.shape[1] > maior[1]:
           maior[1] = a.shape[1]
    return maior

# Função que preenche com 0 pontos faltantes em blocos de hora para ficarem de mesmo tamanho
def formataMovimentosHora(array):
    for i in range(0, len(array)):
        bloco = array[i]
        copia = bloco.copy()
        array[i] = np.zeros(maiorShapeLista(array))
        array[i][:copia.shape[0], :copia.shape[1]] = copia
    return array

def divide_colunas(a, b):
    return a / b if b else 0

# Configuração
tamanho_teste = 5 # Porcentagem do tamanho para teste
path = "dados/dataset.json" # Caminho do dataset vindo do gerador

tamanho_teste /= 100
# Carregamento Dataset Pandas
with open(path , 'r') as f:
    data = json.load(f)
    data = json.loads(data)
    
df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')
##

df["stress"] = df["stress"].astype(int) # True e False para 1 e 0

# Shuffle das horas respeitando a ordem dos pontos
df_naoordenado = df
groups = [df for _, df in df.groupby('date')]
np.random.shuffle(groups)
df = pd.concat(groups).reset_index(drop=True)
##

# Realizar a operação x/y na coordenada
df_treino = df.iloc[:, 0:4]
df_treino.iloc[:,1] = df_treino.iloc[:,1]/df_treino.iloc[:,2]
##

# Normalização das coordenadas para números entre 0 e 1
coordenadas = np.array(df_treino.iloc[:, 1:3]).astype("float32")
normalizador = MinMaxScaler(feature_range=(0,1))
coordenadas = normalizador.fit_transform(coordenadas)
joblib.dump(normalizador, "dados/normalizador.save")
##

# Substituição no dataframe das coordenadas não normalizadas por normalizadas
coordenadas = pd.DataFrame(coordenadas)   
df_treino["x"] = coordenadas[0]
df_treino["y"] = coordenadas[1]
##

# Transforma o dataset plano, em uma lista de vetores numpy (horas)
x = np.split(np.array(df_treino)[:, 1:4].astype("float32"), np.cumsum(np.unique(np.array(df_treino)[:, 0], return_counts=True)[1])[:-1])
x = formataMovimentosHora(x) # Preenche com 0 pontos faltantes em blocos de hora para ficarem de mesmo tamanho
##

# Stress sai do X, para ir para as classes representando o stress de cada bloco de hora
y = pd.DataFrame([], columns=["stress"])
for i in range(0, len(x)):
    bloco = x[i]
    novo = pd.DataFrame(bloco.tolist(), columns=["x","y","stress"]).iloc[:1,2]
    y = pd.concat([y,  pd.DataFrame(novo)])
    x[i] = np.delete(x[i], 2, 1)
    x[i] = np.delete(x[i], 1, 1)
    x[i] = x[i].flatten()
##

# Conversão de tipos numpy para previsores e classe (x,y)
y = np.array(y).astype('int32').flatten()  
x = np.array(x).astype('float32')
##

# Recorte do dataset de testes, retirado x% do treino para ficar isolado
if(tamanho_teste):
    qtdGruposTeste = round(x.shape[0]*tamanho_teste)
    
    x_teste = x[-qtdGruposTeste:][:][:]
    x = x[:-qtdGruposTeste]
    
    y_teste = y[-qtdGruposTeste:]
    y = y[:-qtdGruposTeste]
    
    np.save("dados/x_teste.npy", x_teste)
    np.save("dados/y_teste.npy", y_teste)
##

# Salvamento dos previsores (x) e classe (y) tratados
np.save("dados/x_treino.npy", x)
np.save("dados/y_treino.npy", y)
##

