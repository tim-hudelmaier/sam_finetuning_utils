# Working Notes for finetuning Micro-SAM models

<!--toc:start-->

- [Working Notes for finetuning Micro-SAM models](#working-notes-for-finetuning-micro-sam-models)
  - [Running the script](#running-the-script)
  - [Troubleshooting](#troubleshooting)
  <!--toc:end-->

## Running the script

First build the image for use with singularity:

```{bash}
singularity build micro-sam.sif docker://timjhudelmaier/micro-sam:latest
```

Or find a prebuilt iamge at:

```{bash}
/g/saka/Tim/singularity-images/micro-sam.sif
```

Then run the script:

```{bash}
sbatch /scratch/thudelmaier/micro-sam/run_finetuning.sh
```

Your logs can be found by default at
(for some reason slurm writes the python output into the 'err' file.):

```{bash}
/scratch/thudelmaier/micro-sam/logs
```

If you want to customize the logging location just modify the `run_finetuning.sh`
script.

If you want to change the training hyperparameters, copy `default_config.json` and
use your modified version in the finetune.py call in the `run_finetuning.sh` script.

## Troubleshooting

- Running the script fails with "OSError: GL ES 2.0 library not found"
  Solution: adding the following to the container used

  ```{bash}
  sudo apt-get -y update && sudo apt-get install -y libgl1-mesa-dev`
  ```

  source: ![Github issue](https://github.com/nektos/act/issues/2438)
