# Smartcare Machine Learning - Model
## Development Flow
### [Data processing](DataProcessing.py)
The dataset is processed to images and normalized to fit the machine learn model, [Pandas](https://pandas.pydata.org) and [Numpy](https://numpy.org) libraries were used.

### [Convolutional Neural Network Model](Model.ipynb)
![Convolutional neural network model developed](representacao/2x/modelo.png)
<br>
With [Keras](https://keras.io) and Scikit Learn(https://scikit-learn.org) libraries, a CNN model was created to infer by the alzheimer elderly's movimentation whether it is a wandering movement or not.

### [Train analysis](Model.ipynb)
![Gráficos do modelo com o tensorboard](https://i.imgur.com/fqVUBJ3.jpg)

Initially data was analyzed with [Tensorboard](https://www.tensorflow.org/tensorboard?hl=pt-br) and [Matplotlib](https://matplotlib.org) afterwards migration to google colab notebook. By analyzing the train metrics, the model parameters could be tunned empirically.

### [Evaluation and results](Model.ipynb)

With the validation dataset, our trained model was evaluated and revoke, precision and f1 score metrics were calculated to give an overview of the model prediction quality.
