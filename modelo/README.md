# Smartcare Machine Learning - Modelo
## Fluxo do desenvolvimento
### [Manipulação dos dados](manipulacaoDados.py)
Etapa onde os dados recebidos sofrem algumas pequenas adequações para a rede utilizando principalmente [Pandas](https://pandas.pydata.org) e [Numpy](https://numpy.org)

### [Modelo de redes neurais artificiais](modelo.py)
![Modelo de redes neurais recorrentes utilizado](representacao/2x/modelo.png)
Etapa mais técnica referente a deep learning, a construção/manuntenção e treinamento das redes neurais artificiais recorrentes (RNN) utilizando principalmente [Keras](https://keras.io) e [Scikit Learn](https://scikit-learn.org)

### Análise de resultado
![Gráficos do modelo com o tensorboard](https://i.imgur.com/fqVUBJ3.jpg)
Etapa em que são análisados os resultados de treino e validação do modelo, com base nesses dados são feitas mais alterações nos parâmetros da rede utilizando principalmente [Tensorboard](https://www.tensorflow.org/tensorboard?hl=pt-br) e [Matplotlib](https://matplotlib.org)

