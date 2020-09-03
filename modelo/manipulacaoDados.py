import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import MinMaxScaler



path = "dados/dataset.json"
with open(path , 'r') as f:
    data = json.load(f)
    data = json.loads(data)
    
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
df["stress"] = df["stress"].astype(int) # True e False para 1 e 0

df_treino = df.iloc[:, 1:3].values

previsores = []
stress = []



normalizador = MinMaxScaler(feature_range=(0,1))
df_treino = normalizador.fit_transform(df_treino)

df.to_pickle("dados/dataset.df")

