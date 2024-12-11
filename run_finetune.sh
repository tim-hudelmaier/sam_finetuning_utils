#!/bin/bash

#SBATCH -J th-microsam-finetune
#SBATCH -A saka
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu-el8 -C gaming -n 16 --gres=gpu:3090:1 --mem-per-gpu 64283
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=tim.hudelmaier@embl.de
#SBATCH -o /scratch/thudelmaier/logs/finetune/slurm.%N.%j.out
#SBATCH -e /scratch/thudelmaier/logs/finetune/slurm.%N.%j.err

module load CUDA/12.5.0
apptainer exec --nv --bind /scratch:/scratch docker://timjhudelmaier/pixi-micro-sam-th:latest \
  /bin/bash -c "cd /repo && pixi run python /scratch/thudelmaier/micro-sam/sam_finetuning_utils/finetune.py \
  --config_path /scratch/thudelmaier/micro-sam/sam_finetuning_utils/default_config.json \
  --img_dir /scratch/thudelmaier/leica/data/border_imgs \
  --label_dir /scratch/thudelmaier/leica/data/border_segs \
  --output_dir /scratch/thudelmaier/micro-sam/results/border-finetune
  "
