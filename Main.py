import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
import requests
from torch import nn
from d2l import torch as d2l
from keras.datasets import cifar10
from PIL import Image

class ModelCifar():
    """
    Classe para manipulação do dataset CIFAR-10
    - cria conjunto de validação
    - pré-processamento dos dados
    - treinamento do modelo
    - predição de imagens
    - predição do modelo
    
    Args:
        data_dir(str): diretório onde os dados estão salvos
        batch_size(int): tamanho do batch
    """
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def read_csv_labels(self, fname):
        """
        Lê o arquivo fname e retorna um dicionário de nome para rótulo.
        
        Args:
            fname(str): nome do arquivo
            
        Returns:
            dict: dicionário de nome para rótulo
        """
        with open(fname, 'r') as f:
            # Skip the file header line (column name)
            lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))

    def copyfile(self, filename, target_dir):
        """Copy a file into a target directory."""
        """
        Copia um arquivo para um diretório de destino.
        
        Args:
            filename(str): nome do arquivo
            target_dir(str): diretório de destino
        """
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(filename, target_dir)

    def reorg_train_valid(self, data_dir, labels, valid_ratio):
        """
        Divide o conjunto de treinamento em um conjunto de treinamento e um conjunto de validação.
        
        Args:
            data_dir(str): diretório onde os dados estão salvos
            labels(dict): dicionário de nome para rótulo
            valid_ratio(float): proporção do conjunto de validação
            
        Returns:
            int: número de exemplos por classe para o conjunto de validação
        """
        # O número de exemplos da classe com menos exemplos no conjunto de treinamento
        n = collections.Counter(labels.values()).most_common()[-1][1]

        # O número de exemplos por classe para o conjunto de validação
        n_valid_per_label = max(1, math.floor(n * valid_ratio))
        label_count = {}
        for train_file in os.listdir(os.path.join(data_dir, 'train')):
            label = labels[train_file.split('.')[0]]
            fname = os.path.join(data_dir, 'train', train_file)
            
            # Copia para train_valid_test/train_valid com uma subpasta por classe
            self.copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                        'train_valid', label))
            if label not in label_count or label_count[label] < n_valid_per_label:

                # Copia para train_valid_test/valid
                self.copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                            'valid', label))
                label_count[label] = label_count.get(label, 0) + 1
            else:

                # Copia para train_valid_test/train
                self.copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                            'train', label))
        return n_valid_per_label

    def reorg_test(self, data_dir):
        """
        Reorganiza o conjunto de teste para facilitar a leitura.
        
        Args:
            data_dir(str): diretório onde os dados estão salvos
        """
        for test_file in os.listdir(os.path.join(data_dir, 'test')):
            self.copyfile(os.path.join(data_dir, 'test', test_file),
                    os.path.join(data_dir, 'train_valid_test', 'test',
                                'unknown'))

    def reorg_cifar10_data(self, data_dir, valid_ratio):
        """
        Reorganiza o conjunto de dados CIFAR-10.
        
        Args:
            data_dir(str): diretório onde os dados estão salvos
            valid_ratio(float): proporção do conjunto de validação
        """
        labels = self.read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
        self.reorg_train_valid(data_dir, labels, valid_ratio)
        self.reorg_test(data_dir)

    def transform(self):
        """
        Realiza o pré-processamento dos dados.
        """
        transform_train = torchvision.transforms.Compose([
        # Tranforma a imagem em um quadrado de 40 pixels de altura e largura
        torchvision.transforms.Resize(40),

        # Crop aleatório de uma imagem quadrada de 40 pixels de altura e largura para
        # produzir um quadrado pequeno de 0,64 a 1 vezes a área da imagem original
        # e, em seguida, reduza para um quadrado de 32 pixels de altura e largura
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                    ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),

        # Normaliza cada canal da imagem
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])])


        self.transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])])

        # Aplica as transformações nos dados de treinamento e validação
        self.train_ds, self.train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(self.data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
    
        # Aplica as transformações nos dados de validação e teste
        self.valid_ds, self.test_ds = [torchvision.datasets.ImageFolder(
            os.path.join(self.data_dir, 'train_valid_test', folder),
            transform=self.transform_test) for folder in ['valid', 'test']]

        # Cria os iteradores para os dados de treinamento, validação e teste
        self.train_iter, self.train_valid_iter = [torch.utils.data.DataLoader(
        dataset, self.batch_size, shuffle=True, drop_last=True)
        for dataset in (self.train_ds, self.train_valid_ds)]

        self.valid_iter = torch.utils.data.DataLoader(self.valid_ds, self.batch_size, shuffle=False,
                                                drop_last=True)

        self.test_iter = torch.utils.data.DataLoader(self.test_ds, self.batch_size, shuffle=False,
                                            drop_last=False)

    def get_net(self):
        """
        Cria a rede neural ResNet-18.
        """
        num_classes = 10
        net = d2l.resnet18(num_classes, 3)
        return net

    def train(self, net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
            lr_decay):
        """
        Treina o modelo.
        
        Args:
            net(nn.Module): rede neural
            train_iter(torch.utils.data.DataLoader): iterador de treinamento
            valid_iter(torch.utils.data.DataLoader): iterador de validação
            num_epochs(int): número de épocas
            lr(float): taxa de aprendizado
            wd(float): peso decaimento
            devices(list): lista de dispositivos
            lr_period(int): período de decaimento da taxa de aprendizado
            lr_decay(float): decaimento da taxa de aprendizado
        """
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
        num_batches, timer = len(train_iter), d2l.Timer()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'valid acc'])

        if not devices:
            print("No GPU found, using CPU")
            devices = ['cpu']

        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        
        # Cria uma animação para mostrar o progresso do treinamento, mostrando um gráfico com a perda e acurácia do treinamento e a acurácia da validação
        for epoch in range(num_epochs):
            net.train()
            metric = d2l.Accumulator(3)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                l, acc = d2l.train_batch_ch13(net, features, labels,
                                            self.loss, trainer, devices)
                metric.add(l, acc, labels.shape[0])
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[2], metric[1] / metric[2],
                                  None))
            if valid_iter is not None:
                valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
                animator.add(epoch + 1, (None, None, valid_acc))
            scheduler.step()
        if valid_iter is not None:
            print(f'loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}, '
                f'valid acc {valid_acc:.3f}')
        else:
            print(f'loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(devices)}')

    def train_model(self, devices):
        """
        Define as configurações para o treinamento do modelo e chama a função de treinamento.
        
        Args:
            devices(list): lista de dispositivos
        """
        self.loss = nn.CrossEntropyLoss(reduction="none")

        num_epochs, lr, wd = 20, 1e-4, 5e-4
        lr_period, lr_decay, net = 2, 0.9, self.get_net()

        self.train(net, self.train_iter, self.valid_iter, num_epochs, lr, wd, devices, lr_period,
            lr_decay)

        # Salva os parâmetros do modelo para predições futuras
        torch.save(net.state_dict(), 'model_parameters.pth')

    def predict_image(self, devices, path, classes):
      """
        Função para predizer uma única imagem desejada.
        
        Args:
            devices(list): lista de dispositivos
            path(str): caminho da imagem
            classes(dict): dicionário de classes
            
        Returns:
            str: classe predita
      """
      image = Image.open(path)
      
      # Aplica as transformações na imagem
      tranform_img = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010])])
      transformed_image = tranform_img(image).unsqueeze(0)

      net = self.get_net()
      # Carrega os parâmetros do modelo
      net.load_state_dict(torch.load('model_parameters.pth', map_location=torch.device('cpu')))
      net.eval()
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      net.to(devices[0])
      pred = []

      y_hat = net(transformed_image.to(devices[0]))
      #y_hat = net(X)
      pred.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

      return classes[int(pred[0])]

    def predict_model(self, devices):
        """
        Prediz as classes das imagens do conjunto de teste.
        
        Args:
            devices(list): lista de dispositivos
            
        Returns:
            pd.DataFrame: dataframe com os ids e classes preditas
        """
        net = self.get_net()
        net.load_state_dict(torch.load('model_parameters.pth', map_location=torch.device('cpu')))
        net.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(devices[0])
        preds = []

        for X, _ in self.test_iter:
            y_hat = net(X.to(devices[0]))
            #y_hat = net(X)
            preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
        sorted_ids = list(range(1, len(self.test_ds) + 1))
        sorted_ids.sort(key=lambda x: str(x))
        df = pd.DataFrame({'id': sorted_ids, 'label': preds})
        df['label'] = df['label'].apply(lambda x: self.train_valid_ds.classes[x])
        return df
    
class Main:
  """
  Classe principal para manipulação do dataset CIFAR-10.
  """
  def __init__(self):
    pass

  def create_labels(self, y_train):
    """
    Criar um dataframe com os ids e classes de cada imagem.
    
    Args:
        y_train(list): lista de classes
    
    Returns:
        pd.DataFrame: dataframe com os ids e classes
    """
    labels = [{'id': i+1, 'label': self.classes[y_train[i][0]]} for i in range(len(y_train))]
    return pd.DataFrame(labels)

  def create_dirs(self, data_dir, x_train, y_train, x_test):
    """
    Cria diretórios para salvar os dados de treinamento e teste do CIFAR-10.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Salvar os dados de treinamento e teste
    for i in range(len(x_train)):
        fname = os.path.join(train_dir, f"{i+1}.png")
        torchvision.transforms.functional.to_pil_image(x_train[i]).save(fname)

    # Salvar os rótulos das imagens de treinamento
    train_labels = self.create_labels(y_train)
    train_labels.to_csv(os.path.join(data_dir, "trainLabels.csv"), header=True, encoding='utf-8', index=False)

    for i in range(len(x_test)):
        fname = os.path.join(test_dir, f"{i+1}.png")
        torchvision.transforms.functional.to_pil_image(x_test[i]).save(fname)

  def main(self, pred_image_path, train_test):
    """
    Para tornar o processo mais dinâmico, as pastas com os dados de treinamento e teste do CIFAR-10 são criadas diretamente no código. 
    Assim, não é necessário baixar o dataset manualmente.
    
    Args:
        pred_image_path(str): caminho da imagem a ser predita
        train_test(bool): flag para treinar o modelo ou predizer uma imagem. 
                            Para a API é necessário apenas predizer a imagem. Por isso, essa variável torna o processo mais rápido.
        
    Returns:
        str: classe predita da imagem
    """
    self.classes = {0: 'airplane', 1: 'automobile', 2: 'bird ', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9:'truck'}

    if train_test:
      os.makedirs("cifar-10", exist_ok=True)

    data_dir = 'cifar-10/'

    batch_size = 128
    valid_ratio = 0.1
    devices = d2l.try_all_gpus()

    cifar = ModelCifar(data_dir, batch_size)

    if train_test:
      # Leitura dos dados e criação dos diretórios
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      if not os.listdir(data_dir):
        self.create_dirs(data_dir, x_train, y_train, x_test)
        cifar.reorg_cifar10_data(data_dir, valid_ratio)

      cifar.transform()

      # o modelo só será treinado se não existir um arquivo com os parâmetros salvos
      if not os.path.isfile("model_parameters.pth"):
        cifar.train_model(devices)

      df = cifar.predict_model(devices)
      pd.display(df)
      return
    return cifar.predict_image(devices, pred_image_path, self.classes)

if __name__ == '__main__':
    """
    Chamada da api para predição de uma imagem.
    """
    
    url = 'http://127.0.0.1:5000/predict'

    # alterar o caminho da imagem que deseja predizer
    img = "./imgs_teste/101.png"

    response = requests.post(url, json={
        'img_path': img
    })

    print(response.json())

    im = Image.open(img)
    im.show()