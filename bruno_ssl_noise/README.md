# bruno_ssl_noise

Pipeline de **classificação supervisionada + detecção de label noise** para o
dataset *Cassava Leaf Disease Classification* (Kaggle, 5 classes, ~21k imagens
com rótulos ruidosos vindos de app de campo).

A arquitetura é **inspirada** no artigo *Large-Scale Fully-Unsupervised
Re-Identification* (Bertocco et al., IEEE TBIOM 2025). Adaptamos dois conceitos
do artigo (pré-treino self-supervised de backbone e ensemble de arquiteturas
diversas) para um problema de **classificação supervisionada com rótulos
ruidosos** — que é diferente do ReID totalmente não supervisionado do artigo.
Veja `RELATORIO.md` para a discussão completa do que migrou e do que não migrou.

## Ideia em uma frase

1. **Pré-treino self-supervised (Barlow Twins)** no próprio Cassava para dar um
   bom ponto de partida ao classificador e um espaço de *embeddings* útil.
2. **Fine-tuning supervisionado** de 2-3 backbones diferentes (ResNet50,
   DenseNet121, EfficientNet-B0) e **ensemble** por média de probabilidades.
3. **Detecção de rótulos suspeitos** por três métodos independentes
   (Confident Learning, kNN-consistency, loss-ranking) e comparação da
   concordância entre eles.

## Estrutura

```
bruno_ssl_noise/
├── configs/            # YAMLs de configuração de experimentos
├── data/               # dataset, transforms, gerador de dados dummy
├── models/             # factory de backbones + cabeça Barlow Twins
├── core/               # os 3 detectores de ruído + concordância
├── notebooks/          # SÓ visualização pós-treino
├── ssl_pretrain.py     # Etapa 1: Barlow Twins
├── train.py            # Etapa 2: fine-tuning supervisionado (1 backbone por run)
├── evaluate.py         # Etapa 3: métricas por backbone + ensemble
├── detect_noise.py     # Etapa 4: roda os 3 detectores e mede concordância
├── utils.py            # métricas, embeddings, checkpoint, seed
├── run_all.sh          # exemplo de pipeline ponta a ponta (SLURM-friendly)
└── RELATORIO.md        # relatório do trabalho
```

## Uso rápido (dados reais)

```bash
# 0. baixar o dataset do Kaggle para $DATA (train.csv + train_images/)

# 1. pré-treino self-supervised
python ssl_pretrain.py --config configs/default.yaml --data_dir $DATA \
    --out_dir runs/ssl

# 2. treinar cada backbone (o ensemble é só treinar 3 vezes trocando --model)
python train.py --config configs/default.yaml --data_dir $DATA \
    --model resnet50 --ssl_backbone runs/ssl/backbone.pth --out_dir runs/resnet50
python train.py --config configs/default.yaml --data_dir $DATA \
    --model densenet121 --ssl_backbone runs/ssl/backbone.pth --out_dir runs/densenet121
python train.py --config configs/default.yaml --data_dir $DATA \
    --model efficientnet_b0 --ssl_backbone runs/ssl/backbone.pth --out_dir runs/effnet

# 3. avaliar cada um E o ensemble
python evaluate.py --data_dir $DATA \
    --checkpoints runs/resnet50/best.pth runs/densenet121/best.pth runs/effnet/best.pth \
    --out_dir runs/eval

# 4. detectar rótulos suspeitos (usa as predições/embeddings gerados acima)
python detect_noise.py --data_dir $DATA --eval_dir runs/eval \
    --ssl_backbone runs/ssl/backbone.pth --out_dir runs/noise
```

## Smoke-test (sem dataset real)

Gera um dataset sintético pequeno e roda o pipeline inteiro em CPU só para
confirmar que cada script executa e encadeia:

```bash
python data/make_dummy.py --out_dir /tmp/cassava_dummy --n 60
bash run_all.sh /tmp/cassava_dummy runs_smoke --smoke
```
