import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import CassavaDataset

# 1. Define as transformações (Data Augmentation)
train_transform = A.Compose([
    A.Resize(256, 256), # Redimensiona a imagem
    A.RandomCrop(224, 224), # Recorte aleatório para os backbones modernos
    A.HorizontalFlip(p=0.5), # Inverte a folha
    A.RandomBrightnessContrast(p=0.2), # Ajuda com a iluminação ruim
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Padrão ImageNet
    ToTensorV2(), # Converte para Tensor do PyTorch
])

# 2. Instancia o dataset
train_dataset = CassavaDataset(
    csv_file="cassava_data/train.csv",
    img_dir="cassava_data/train_images",
    transform=train_transform
)

# 3. Cria o DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, # Embaralha os dados a cada época
    num_workers=2 # Usa múltiplos núcleos da CPU para ler do disco mais rápido
)

# 4. Simulação de como isso entra na rede durante o loop de treino
for imagens, labels in train_loader:
    # Manda os dados para a GPU
    imagens = imagens.cuda()
    labels = labels.cuda()
    
    # Passa as imagens no seu modelo (ResNet, ConvNeXt, etc)
    # predicoes = modelo(imagens)
    # erro = funcao_de_perda(predicoes, labels)
    # erro.backward()
    # otimizador.step()
    
    break # Apenas para não rodar o loop inteiro neste exemplo