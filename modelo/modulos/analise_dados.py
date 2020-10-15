from time import time
from contextlib import redirect_stdout
from sklearn.utils import shuffle
from numpy import mean, std, array, rint
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from modulos.machine_learning import kfold
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import os



def kfold_log_batches(model_func, x, y, num_folds=10, checks_per_batch=5, batches=[]):
    if not batches:
        batches = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130]
    model = model_func()
    histories = []
    identificador = int(time())
    f = open("logs/batch_size/batch-size_%s.log"%(identificador), "a")
    
    with redirect_stdout(f):
        model.summary()
        
    f.write('\n'+str(model.optimizer.get_config()))
    f.close()
    
    
    for batch in batches:
        medias = []
        desvios_padrao = []
        f = open("logs/batch_size/batch-size_%s.log"%(identificador), "a")
        print("Batch Size: %d:"%(batch))
        f.write("\nBatch Size %d: "%(batch))
        f.close()
        
        for i in range(0, checks_per_batch):
            cvscores = []
            
            f = open("logs/batch_size/batch-size_%s.log"%(identificador), "a")
            h, cvscores = kfold(x, y, model, batch_size=batch, verbose=0)
            f.write(" %.2f%% (+/- %.2f%%); " % (mean(cvscores), std(cvscores)))
            histories.append(h)
            medias.append(mean(cvscores))
            desvios_padrao.append(std(cvscores))
            f.close()
            print("Batch %s - Check %d: %.2f%% (+/- %.2f%%)" % (batch, i+1, mean(cvscores), std(cvscores))) # Resultado Final
            ##
        f = open("logs/batch_size/batch-size_%s.log"%(identificador), "a")
        f.write(" Média: %.2f%% (+/- %.2f%%)"%(mean(medias), mean(desvios_padrao)))
        f.close()
    return histories

# Função que plota um gráfico comparativo entre as classes reais e previstas
def plotarComparacoes(model, x_teste, y_teste, estatisticas=1):
    previsao = model.predict(x_teste)
    previsao = array([previsao > 0.5]).astype('int32')
    previsao = previsao.reshape(previsao.shape[1])
    
    if estatisticas:
        sm = SequenceMatcher(None,previsao, y_teste)
        print("\nEstatísticas")
        print("\tMédia das previsões: %.2f"%(previsao.mean()))
        print("\tMédia real: %.2f"%(y_teste.mean()))
        print("\tDesvio Padrão das previsões: %.2f"%(previsao.std()))
        print("\tDesvio Padrão: %.2f"%(y_teste.std()))
        print("\tPorcentagem de similaridade: %.2f %%"%(sm.ratio()*100))
    
    plt.plot(previsao, color = 'blue', label = 'Previsão', alpha=0.5)
    plt.plot(y_teste, color = 'red', label = 'Real', alpha=0.5)
    plt.title('Previsão de stress de idoso com base na movimentação')
    plt.xlabel('Intervalo de tempo')
    plt.ylabel('Stress')
    plt.legend()
    plt.show()
    
def plotar(h):
    if not isinstance(h, list):
        plt.plot(h.history['accuracy'])
        if 'val_accuracy' in h.history:
            plt.plot(h.history['val_accuracy'])
        plt.title('AcurÃ¡cia do modelo')
        plt.ylabel('acurÃ¡cia')
        plt.xlabel('Ã©poca')
        plt.legend(['Treino', 'Teste'], loc='best')
        plt_accuracy = plt.gcf()
        plt.show()  
        
        plt.plot(h.history['loss'])
        if 'val_loss' in h.history:
            plt.plot(h.history['val_loss'])
        plt.title('Perda do modelo')
        plt.ylabel('perda')
        plt.xlabel('Ã©poca')
        plt.legend(['train', 'test'], loc='best')
        plt_loss = plt.gcf()
        plt.show()
        return plt_loss, plt_accuracy

    for historico in range(h):
        plt.plot(historico.history['accuracy'])
        if 'val_accuracy' in historico.history:
            plt.plot(historico.history['val_accuracy'])
        plt.title('AcurÃ¡cia do modelo')
        plt.ylabel('acurÃ¡cia')
        plt.xlabel('Ã©poca')
        plt.legend(['Treino', 'Teste'], loc='best')
        plt.show()  
        plt.plot(historico.history['loss'])
        if 'val_loss' in historico.history:
            plt.plot(historico.history['val_loss'])
        plt.title('Perda do modelo')
        plt.ylabel('perda')
        plt.xlabel('Ã©poca')
        plt.legend(['train', 'test'], loc='best')
        plt.show()
    return 1

def plotar_log(lista, tags, nfolds=10):
    path = "logs/plots/%d"%(int(time()))
    os.mkdir(path)
    print("Total Lista: %d | %d lista[0]"%(len(lista), len(lista[0])))  
    i=0
    for grupo in lista:
        j=0
        fold=0
        print(fold)
        for h in grupo:
            
            if i%5 == 0:
                tag = tags[j]
                j = j + 1
            if not os.path.exists("%s/%d"%(path, tag)):
                os.mkdir("%s/%d"%(path, tag))
            plot_accuracy, plot_loss = plotar(h)

            print(i%nfolds)    
            if i%(nfolds/len(tags)) == 0:
                if not os.path.exists("%s/%d/%d"%(path, tag, fold)):
                    os.mkdir("%s/%d/%d"%(path, tag, fold))
                    fold = fold + 1
            
            print("%s/%d/%d/%d_accuracy.png"%(path, tag, fold-1, i))
            plot_accuracy.savefig("%s/%d/%d/%d_accuracy.png"%(path, tag, fold-1, i))
            plot_loss.savefig("%s/%d/%d/%d_loss.png"%(path, tag, fold-1, i))
            i = i+1

def matriz_confusao(y_teste, y_previsto):
     cm = confusion_matrix(y_teste, y_previsto)  
     plot_confusion_matrix(cm, class_names=["Estressado", "Não Estressado"])
     plt.show()
def plotar_matriz_confusao(model, x_teste, y_teste):
    y_pred = rint(array(model.predict(x_teste)))
    matriz_confusao(y_teste, y_pred)
    
    