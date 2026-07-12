# Prompt de handoff — rodar o projeto Cassava nas A100

> Copie o texto abaixo e mande para o agente com acesso remoto ao cluster.

---

Você tem acesso shell a um cluster SLURM com GPUs NVIDIA A100 (80GB). Preciso que
você rode um projeto de deep learning já pronto, do começo ao fim, numa A100, e me
reporte os resultados. **O código já está implementado e validado — não reescreva
nada; sua tarefa é executar, monitorar e reportar.** Só faça alterações se algo
quebrar, e nesse caso me explique o que mudou e por quê.

## O que é o projeto

Classificação de doenças em folhas de mandioca (dataset *Cassava Leaf Disease
Classification*, Kaggle 2020: ~21.397 imagens, 5 classes, rótulos ruidosos) com
dois objetivos: (1) classificação supervisionada usando pré-treino
self-supervised (Barlow Twins) + ensemble de 3 backbones; (2) detecção de
rótulos suspeitos por 3 métodos independentes (Confident Learning/cleanlab,
kNN-consistency no espaço SSL, loss-ranking) e a concordância entre eles.

A arquitetura é adaptada do artigo *Large-Scale Fully-Unsupervised
Re-Identification* (Bertocco et al., TBIOM 2025). Detalhes completos estão no
`RELATORIO.md` da pasta.

## Estrutura da pasta `bruno_ssl_noise/`

```
configs/default.yaml     # hiperparâmetros (pode sobrescrever por CLI)
data/                    # dataset, transforms, geradores de amostra
models/                  # factory de backbones + cabeça Barlow Twins
core/                    # os 3 detectores de ruído + concordância
ssl_pretrain.py          # etapa 1: Barlow Twins
train.py                 # etapa 2: fine-tuning (1 backbone por run)
evaluate.py              # etapa 3: métricas por backbone + ensemble
detect_noise.py          # etapa 4: detecção de ruído + concordância
run_slurm.sbatch         # pipeline completo pronto p/ SLURM
requirements.txt
```

## Passos

1. **Ambiente.** Crie um venv/conda (Python 3.10+), ative e instale:
   `pip install -r requirements.txt`. Confirme que o PyTorch enxerga a GPU:
   `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
   O cluster tem internet, então os pesos ImageNet (via timm/HuggingFace) e os
   backbones baixam normalmente — **não** use a flag `--no_pretrained`.

2. **Dados.** Descompacte o dataset da competição 2020 numa pasta `DATA` que
   contenha `train.csv` (colunas `image_id,label`, labels 0–4) e
   `train_images/`. Se o zip `cassava-leaf-disease-classification.zip` estiver
   disponível: `unzip -q cassava-leaf-disease-classification.zip -d DATA`
   (pode ignorar `train_tfrecords/`/`test_*`).

3. **Selecione uma A100.** Descubra como o cluster expõe as A100
   (`sinfo -o "%P %G %N"` mostra partições e GRES) e ajuste o topo do
   `run_slurm.sbatch` com a partição/constraint corretas (ex.:
   `#SBATCH --partition=<fila_a100>` ou `#SBATCH --constraint=a100`).
   Se as A100 estiverem todas ocupadas, uma **L40S (48GB)** ou **RTX A6000
   (48GB)** servem igualmente bem — me avise se cair numa dessas.

4. **Rode o pipeline completo:**
   ```bash
   sbatch run_slurm.sbatch /caminho/para/DATA
   tail -f slurm_*.out
   ```
   Isso executa, em sequência: SSL (Barlow Twins, resnet50) → fine-tuning de
   resnet50, densenet121 e efficientnet_b0 → avaliação individual + ensemble →
   detecção de ruído. Tempo esperado numa A100: ~3–4h (o SSL é a etapa mais
   longa). Se quiser uma primeira rodada mais rápida (~2h), acrescente
   `--epochs 40` na linha do `ssl_pretrain.py` dentro do `.sbatch`.

5. **Me reporte, ao final:**
   - `runs/eval/summary.csv` — acurácia e macro-F1 de cada backbone e do ensemble.
   - `runs/eval/ensemble_report.json` — F1 por classe + matriz de confusão.
   - `runs/noise/agreement.json` e `runs/noise/votes.csv` — quantas imagens
     suspeitas cada método achou e quantas foram apontadas por ≥2 métodos.
   - Qualquer erro/warning relevante do `slurm_*.out`.

## Coisas importantes (para não "consertar" errado)

- **O SSL é feito só para o backbone primário (resnet50)**, que também vira o
  espaço de embeddings do kNN de ruído. Os outros membros do ensemble
  (densenet121, efficientnet_b0) usam init ImageNet. Isso é intencional — não
  tente carregar o backbone SSL nas outras arquiteturas (dá erro de shape, e o
  código já ignora isso de propósito).
- **Limitação conhecida da detecção de ruído:** Confident Learning e
  kNN operam no split de validação; loss-ranking opera no treino. Então a
  concordância entre loss-ranking e os outros dois fica subestimada (universos
  de imagens diferentes). Está documentado; não é bug. Se sobrar tempo e você
  quiser melhorar, o caminho é validação cruzada k-fold para gerar
  probabilidades out-of-fold de todas as imagens e rodar os 3 métodos sobre o
  mesmo conjunto.
- **AMP (mixed precision) ainda não está implementado** (treino é FP32). Se
  quiser acelerar ~1.5–2×, dá para envolver os forward/backward de `train.py` e
  `ssl_pretrain.py` com `torch.cuda.amp.autocast` + `GradScaler` — mas só faça
  isso se eu pedir; a prioridade é rodar como está primeiro.
- Os scripts salvam checkpoint a cada N épocas (config `train.ckpt_every`), então
  jobs interrompidos podem ser retomados a partir do último checkpoint.

Ao terminar, me mande um resumo curto: os números do `summary.csv`, quantas
amostras suspeitas foram encontradas e a taxa de concordância entre os 3 métodos.
