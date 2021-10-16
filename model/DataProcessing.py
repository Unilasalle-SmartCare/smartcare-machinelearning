#!/usr/bin/env python
# coding: utf-8

# Abrir python 3.5
# In[2]:


import pandas as pd
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


path = "dados/eval.json" # Caminho do dataset vindo do gerador
# Carregamento Dataset Pandas
with open(path , 'r') as f:
    data = json.load(f)
    data = json.loads(data)


# In[5]:


df = pd.DataFrame.from_dict(pd.io.json.json_normalize(data), orient='columns')
##
df["stress"] = df["stress"].astype(int) # True e False para 1 e 0


# In[7]:



# https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays
def naive_line(r0, c0, r1, c1):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = naive_line(c0, r0, c1, r1)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return naive_line(r1, c1, r0, c0)

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

    valbot = np.floor(y)-y+1
    valtop = y-np.floor(y)

    return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),
            np.concatenate((valbot, valtop)))


# In[50]:


# split the array into daily paths
x = np.split(np.array(df)[:, 1:4].astype("int"), np.cumsum(np.unique(np.array(df)[:, 0], return_counts=True)[1])[:-1])
# create a list of 2D arrays of 602x369
pathImgs = np.zeros((len(x),602,369))

y = df.groupby(by=["date"]).first()["stress"].values

# In[51]:


# For each 2D array, paint the path that was made in that day.
for pathIdx,currPath in enumerate(x):
    prevRow =  currPath[0]
    for i,row in enumerate(currPath):
        pathImgs[pathIdx,row[1],row[2]] = 1
        if i > 0:
            #draw a line between the previous xy point and the current xy point
            rr, cc, _ = naive_line(prevRow[1],prevRow[2], row[1], row[2])
            ## rr and cc are a lists of indexes that defines where in the x,y axis we our line is drawn
            pathImgs[pathIdx,rr, cc] = 1
        prevRow = row


# In[59]:
np.save("dados/x-validacaoo.npy", pathImgs)
np.save("dados/y-validacaoo.npy", y)

for i,img in enumerate(pathImgs):
    plt.title("") 
    plt.imshow(img.T, cmap="gray")
    plt.show()

