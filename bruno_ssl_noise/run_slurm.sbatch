#!/usr/bin/env bash
#SBATCH --job-name=cassava_ssl_noise
#SBATCH --gres=gpu:1              # 1 GPU (Titan RTX 24GB ou A100 80GB)
#SBATCH --cpus-per-task=8         # workers do DataLoader (alimenta a GPU)
#SBATCH --mem=32G
#SBATCH --time=24:00:00           # margem folgada; ajuste ao seu cluster
#SBATCH --output=slurm_%j.out     # log (stdout+stderr) com o número do job
#SBATCH --error=slurm_%j.out
#
# Pipeline completo no cluster: SSL -> 3 backbones -> avaliação -> detecção de ruído.
#
# Uso:
#   sbatch run_slurm.sbatch /caminho/do/cassava
# onde a pasta tem train.csv e train_images/ (dataset da competição 2020).
#
# Para acompanhar o progresso:
#   tail -f slurm_<JOBID>.out

set -e

DATA=${1:?informe o caminho do dataset (pasta com train.csv e train_images/)}
RUNS=${2:-runs}          # onde salvar checkpoints e resultados
CFG=configs/default.yaml

# --- ambiente (ajuste ao seu cluster) ---
# module load cuda/12.1
# source ~/venvs/cassava/bin/activate
echo "GPU visível:"; nvidia-smi -L || true
echo "Início: $(date)  | DATA=$DATA  | RUNS=$RUNS"

MODELS=("resnet50" "densenet121" "efficientnet_b0")
SSL_MODEL="resnet50"     # membro primário: ganha o SSL e vira o espaço do kNN

# ===== [1/4] Pré-treino self-supervised (Barlow Twins) =====
# 100 épocas é o padrão do config. Para uma primeira rodada mais rápida,
# acrescente --epochs 40 (costuma dar quase todo o ganho).
echo "==> [1/4] SSL pretrain ($SSL_MODEL)"
python ssl_pretrain.py --config $CFG --data_dir "$DATA" --model "$SSL_MODEL" \
    --out_dir "$RUNS/ssl" --num_workers 8

# ===== [2/4] Fine-tuning supervisionado (um run por backbone) =====
CKPTS=()
for M in "${MODELS[@]}"; do
    echo "==> [2/4] treino $M"
    python train.py --config $CFG --data_dir "$DATA" --model "$M" \
        --ssl_backbone "$RUNS/ssl/backbone.pth" --out_dir "$RUNS/$M"
    CKPTS+=("$RUNS/$M/best.pth")
done

# ===== [3/4] Avaliação individual + ensemble =====
echo "==> [3/4] avaliação"
python evaluate.py --config $CFG --data_dir "$DATA" \
    --checkpoints "${CKPTS[@]}" --out_dir "$RUNS/eval"

# ===== [4/4] Detecção de rótulos suspeitos =====
echo "==> [4/4] detecção de ruído"
python detect_noise.py --config $CFG --data_dir "$DATA" \
    --eval_dir "$RUNS/eval" --pred_prefix ensemble \
    --ssl_backbone "$RUNS/ssl/backbone.pth" --train_dir "$RUNS/resnet50" \
    --out_dir "$RUNS/noise"

echo "Fim: $(date). Resultados em $RUNS/  (summary.csv, noise/votes.csv, etc.)"
