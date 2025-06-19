#!/bin/bash
#SBATCH --job-name=aml_training
#SBATCH --partition=st,general
#SBATCH --qos=medium
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#SBATCH --error=slurm_errors/slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END

CONTAINER="aml_container.sif"
OVERLAY="storage_overlay.img"

#TODO: change to your results path
RESULTPATH="/results:/results" 

#TODO: change to your data path
DATAPATH= "/data:/data" 

# Run command inside the container
srun /usr/bin/apptainer run --nv -B "$RESULTPATH" -B "$DATAPATH" $CONTAINER \
  --model interleaved \  # Change to your desired model
  --data Small_HI \ # Change to your desired dataset
  --n_epochs 60 \
  --batch_size 256 \
  --save_model \
  --emlps
