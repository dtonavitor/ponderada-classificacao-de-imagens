# Atividade: Classificação de Imagens (CIFAR-10)

## Descrição do dataset e da implementação
### O dataset
O CIFAR-10 é um conjunto de dados de visão computacional estabelecido, usado para reconhecimento de objetos. Ele é um subconjunto do conjunto de dados de 80 milhões de imagens minúsculas e consiste em 60.000 imagens coloridas 32x32 contendo uma das 10 classes de objetos, com 6.000 imagens por classe. Há 50.000 imagens de treinamento e 10.000 imagens de teste nos dados oficiais.

As classes no conjunto de dados são:
- avião 
- automóvel 
- pássaro 
- gato 
- veado 
- cachorro 
- sapo 
- cavalo 
- navio 
- caminhão

### Implementação
A implementação foi feita seguindo o seguinte tutorial: https://pt.d2l.ai/chapter_computer-vision/kaggle-cifar10.html. Aqui é utilizada a rede ResNet18. A ResNet18 é uma rede neural convolucional (CNN) profunda com 18 camadas, projetada para tarefas de classificação de imagens. Ela faz parte da família de redes residuais (ResNets), introduzidas por Kaiming He e seus colegas em 2015, com o objetivo de enfrentar o problema da perda do gradiente e de treinar redes neurais muito profundas de maneira eficiente. O elemento chave da ResNet é o bloco residual, que permite a criação de conexões de atalho (skip connections) que ignoram uma ou mais camadas. Estas conexões diretas ajudam a mitigar o problema do gradiente, facilitando a propagação dos gradientes durante o treinamento.

A ResNet18 tem quatro estágios principais, cada um contendo dois blocos residuais. Cada bloco é composto por:

- Duas camadas convolucionais de tamanho 3x3.
- Função de ativação ReLU após cada convolução.
- Uma conexão de atalho que adiciona a entrada do bloco diretamente à saída, antes da aplicação da função de ativação final.

#### Modificações e configuração do repositório
Neste repositório é possível encontrar alguns arquivos:

- `requirements.txt`: lista de todas as dependências necessárias para a execução do código. Para instalá-las execute ```pip install -r requirements.txt```.

- `/imgs_teste`: imagens .png utilizadas para casos de teste da classificação

- `model_parameters.pth`: arquivo com os parâmetros do modelo treinado. Com ele é possível realizar apenas a predição.
    * este modelo foi treinado:
      * 20 épocas
      * learing rate: 1e-4
      * learning rate decay: 0.9
      * weight decay: 5e-4
      * learning rate period: 2
      * loss: CrossEntropyLoss
      * optimizer: SGD
      * batch_size: 128

- `app.py`: código do servidor Flask responsável pela predição de uma imagem. Recebe como corpo da requisição o caminho da imagem.

- `Main.py`: arquivo principal com as seguintes especificidades:
    * classe `ModelCifar()`: responsável pelo pré-processamento do dataset, treino e predição do modelo.
    * classe `Main()`: responsável pelo download e organização do dataset (não é necessário o download manual) e chamada dos métodos da `ModelCifar()`. É possível executá-la para treinamento e predição do dataset de teste ou apenas para a classificação de uma imagem por meio do parâmetro booleano train_test (a api chamará essa classe com train_test = False)
    * quando executado via `python Main.py` é feita a requisição para a API e imprimido no terminal o resultado da predição como um json (formato `{classe: ''}`)

Para executar a predição:
1. instale as dependências
2. execute o servidor com `python app.py`
3. em `Main.py` altere o caminho da imagem que deseja predizer
    ```
    Linha 410
    img = "altere aqui"
    ```
4. execute a requisição com `python Main.py` em outro terminal

Também é possível executar via o google colab: https://colab.research.google.com/drive/1fQcnza25P2vzFmYmpfPuZ_Pc8Y61VYRR?usp=sharing. Especificamente para o treinamento é mais desejável, pois é possível utilizar GPU que torna mais o processo mais rápido.

## Resultados
Os testes foram feitos com imagens do próprio dataset, que não foram usadas nem para treino ou validação.
### Teste 1:

imagem de um gato 

![Imagem de um gato](./imgs_teste/1.png)

resultado do modelo: {'classe': 'cat'}

### Teste 2:

imagem de um navio

![Imagem de um navio](./imgs_teste/1000.png)

resultado do modelo: {'classe': 'ship'}

### Teste 3:

imagem de um veado

![Imagem de um navio](./imgs_teste/101.png)

resultado do modelo: {'classe': 'deer'}