#!/bin/bash

#SBATCH -J th-microsam-finetune
#SBATCH -A saka
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu-el8 -C gaming -n 16 --gres=gpu:3090:1 --mem-per-gpu 64283
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=tim.hudelmaier@embl.de
#SBATCH -o /scratch/thudelmaier/micro-sam/logs/slurm.%N.%j.out
#SBATCH -e /scratch/thudelmaier/micro-sam/logs/slurm.%N.%j.err

module load CUDA/12.5.0
singularity exec --nv --bind /scratch/thudelmaier/micro-sam:/scratch/thudelmaier/micro-sam micro-sam.sif \
  bash -c "source activate micro-sam && \
  python /scratch/thudelmaier/micro-sam/finetune.py \
  --config_path /scratch/thudelmaier/micro-sam/default_config.json \
  --img_dir /scratch/thudelmaier/micro-sam/avia \
  --label_dir /scratch/thudelmaier/micro-sam/segmentation \
  --output_dir /scratch/thudelmaier/micro-sam/results/test-finetune
  "
