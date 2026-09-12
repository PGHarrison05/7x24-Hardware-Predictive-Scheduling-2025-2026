#!/bin/bash
#SBATCH --job-name=ai_sim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

mkdir -p logs scratch

python3 ai_workload_sim.py \
  --job-name "rigid_short_high" \
  --flexibility rigid \
  --length short \
  --stress high \
  --mem-mb 1024 \
  --io-mb 50 \
  --checkpoint-mb 0 \
  --checkpoint-interval 0 \
  --scratch-dir ./scratch \
  --log-file logs/workload_log.jsonl
