#!/bin/bash
#SBATCH --job-name=ddp-test1
#SBATCH --partition=gpu
#SBATCH --time=9:00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=1792
#SBATCH --chdir=[PATH_TO_THE_MAIN_FOLDER]
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=9304
export WORLD_SIZE=6

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
### Must be updated with your environment

source ~/.bashrc

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh

conda activate [YOUR_CONDA_ENV]

# conda activate flat-samples

srun python train_ddp_test.py


