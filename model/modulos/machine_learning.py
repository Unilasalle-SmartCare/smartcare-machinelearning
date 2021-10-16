import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, train_test_split

def carrega_dados():
    # Dados de treino
    x = np.load("dados/x_treino.npy")
    y = np.load("dados/y_treino.npy")
    ##
    x, x_teste, y, y_teste = train_test_split(x, y)
    
    x,y = shuffle(x,y)
    x_teste, y_teste = shuffle(x_teste,y_teste)
    
    return x, y, x_teste, y_teste

def kfold(x, y, model, kfold_=StratifiedKFold, n=10, epochs=500, batch_size=75, verbose=1):
    x, y = shuffle(x , y)
    kf = kfold_(n_splits=n, shuffle=True)
    cvscores = []
    histories = []
    for treino, teste in kf.split(x, y):
        history = model.fit(x[treino], y[treino], epochs=epochs, batch_size=batch_size, verbose=0)
        scores = model.evaluate(x[teste], y[teste], verbose=0)
        cvscores.append(scores[1] * 100)
        histories.append(history)
    if(verbose):
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) # Resultado Final
    return histories, cvscores
