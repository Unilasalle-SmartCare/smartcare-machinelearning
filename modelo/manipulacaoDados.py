import pandas as pd
import json
from pandas.io.json import json_normalize


path = "dados/dataset.json"
with open(path , 'r') as f:
    data = json.load(f)
    data = json.loads(data)
    
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

df.to_pickle("dados/dataset.df")

