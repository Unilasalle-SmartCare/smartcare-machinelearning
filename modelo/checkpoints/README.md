# Smartcare Machine Learning - Modelo - Checkpoints
Neste diretório são armazenados os checkpoints do keras gerados durante o treinamento

## Por que usar checkpoints ?
Muitas vezes passamos horas treinando o modelo e por algum motivo ele tem que ser encerrado antes do término. Com os checkpoints, o keras armazena o modelo e os melhores pesos (A definição de melhores pesos é configurável, pode ser um menor loss ou maior accuracy) a cada época. Portanto caso o processo de treino seja abruptamente encerrado, é possível carregar um checkpoint e treinar a partir dos melhores pesos salvos como checkpoint anteriormente.
