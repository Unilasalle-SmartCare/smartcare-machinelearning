import pandas as pd
import json
import joblib
from sklearn.preprocessing import MinMaxScaler



path = "dados/dataset.json"
with open(path , 'r') as f:
    data = json.load(f)
    data = json.loads(data)
    
df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')
df["stress"] = df["stress"].astype(int) # True e False para 1 e 0

df_treino = df.iloc[:, 1:4].values


normalizador = MinMaxScaler(feature_range=(0,1))

df_treino = normalizador.fit_transform(df_treino)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(df_treino[:, 1:2])

joblib.dump(normalizador, "dados/normalizador.save")
joblib.dump(normalizador_previsao, "dados/normalizador-previsao.save")



df["x"] = df_treino[:, 0]
df["y"] = df_treino[:, 1]
df["stress"] = df_treino[:, 2]



df.to_pickle("dados/dataset-treino.df")

