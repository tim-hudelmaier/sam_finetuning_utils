#!/bin/bash

#SBATCH -J th-microsam-finetune
#SBATCH -A saka
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu-el8 -C gaming -n 16 --gres=gpu:3090:1 --mem-per-gpu 64283
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=tim.hudelmaier@embl.de
#SBATCH -o /scratch/thudelmaier/logs/auto_segment/slurm.%N.%j.out
#SBATCH -e /scratch/thudelmaier/logs/auto_segment/slurm.%N.%j.err

module load CUDA/12.5.0
apptainer exec --nv --bind /scratch:/scratch docker://timjhudelmaier/pixi-micro-sam-th:latest \
  /bin/bash -c "cd /repo && pixi run python /scratch/thudelmaier/micro-sam/sam_finetuning_utils/automatic_3d_segmentation.py \
  --img_path /scratch/thudelmaier/leica/data/new_data_extracted/unperturbed_scene5.tif \
  --ndim 3 \
  --checkpoint /scratch/thudelmaier/micro-sam/results/clahe-finetune-vit-l-lm/models/checkpoints/vit_l_lm_clahe/best.pt \
  --model_type vit_l_lm \
  --output_path /scratch/thudelmaier/micro-sam/results/new_data/segmentation_unperturbed.tiff \
  --channels_path /scratch/thudelmaier/micro-sam/sam_finetuning_utils/used_channels.txt \
  --clahe False \
  --embeddings_out_path /scratch/thudelmaier/micro-sam/results/new_data/new_data_3d_seg_ActD
  "
