#!/usr/bin/env bash
# Pipeline ponta a ponta. Também serve de referência para um job SLURM
# (cada python ... pode virar um passo separado com dependência).
#
# Uso:
#   bash run_all.sh <DATA_DIR> <RUNS_DIR> [--smoke]
#
# Exemplo real:
#   bash run_all.sh /data/cassava runs
# Smoke-test (rápido, CPU):
#   python data/make_dummy.py --out_dir /tmp/cassava_dummy --n 60
#   bash run_all.sh /tmp/cassava_dummy runs_smoke --smoke

set -e

DATA=${1:?informe o DATA_DIR}
RUNS=${2:?informe o RUNS_DIR}
SMOKE=""
SSL_FLAGS=""
TRAIN_FLAGS=""
# No smoke-test rodamos offline (sem baixar pesos ImageNet do HuggingFace).
if [ "$3" == "--smoke" ]; then SMOKE="--smoke"; SSL_FLAGS="--no_pretrained"; TRAIN_FLAGS="--no_pretrained"; fi

CFG=configs/default.yaml
MODELS=("resnet50" "densenet121" "efficientnet_b0")
SSL_MODEL="resnet50"   # o SSL é feito para 1 arquitetura (o membro "primário")
if [ -n "$SMOKE" ]; then MODELS=("resnet18" "resnet18"); SSL_MODEL="resnet18"; fi  # leve p/ teste

# O backbone SSL casa com o membro primário do ensemble (ganha o pré-treino) e
# também vira o espaço de embeddings do kNN de ruído. Os demais membros do
# ensemble entram com init ImageNet.
echo "==> [1/4] SSL pretrain ($SSL_MODEL)"
python ssl_pretrain.py --config $CFG --data_dir "$DATA" --model "$SSL_MODEL" \
    --out_dir "$RUNS/ssl" $SSL_FLAGS $SMOKE

CKPTS=()
for i in "${!MODELS[@]}"; do
    M=${MODELS[$i]}
    OUT="$RUNS/${M}_$i"
    echo "==> [2/4] treino backbone $M -> $OUT"
    python train.py --config $CFG --data_dir "$DATA" --model "$M" \
        --ssl_backbone "$RUNS/ssl/backbone.pth" --out_dir "$OUT" $TRAIN_FLAGS $SMOKE
    CKPTS+=("$OUT/best.pth")
done

echo "==> [3/4] avaliação individual + ensemble"
python evaluate.py --config $CFG --data_dir "$DATA" \
    --checkpoints "${CKPTS[@]}" --out_dir "$RUNS/eval" $SMOKE

echo "==> [4/4] detecção de ruído"
FIRST_TRAIN="$RUNS/resnet50_0"
if [ -n "$SMOKE" ]; then FIRST_TRAIN="$RUNS/resnet18_0"; fi
python detect_noise.py --config $CFG --data_dir "$DATA" \
    --eval_dir "$RUNS/eval" --pred_prefix ensemble \
    --ssl_backbone "$RUNS/ssl/backbone.pth" --train_dir "$FIRST_TRAIN" \
    --out_dir "$RUNS/noise" $SMOKE

echo "==> pipeline concluído. Resultados em $RUNS/"
