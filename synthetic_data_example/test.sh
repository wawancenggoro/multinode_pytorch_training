#!/bin/bash
#SBATCH --job-name=ddp-test1
#SBATCH --partition=gpu
#SBATCH --time=9:00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1792G
#SBATCH --chdir=/scratch/bmkg1/git/multinode_pytorch_training/synthetic_data_example/
#SBATCH --output=slurm_monitoring/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=9304
export WORLD_SIZE=24

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
### Must be updated with your environment

source ~/.bashrc

### activate conda env
source /scratch/bmkg1/miniconda3/etc/profile.d/conda.sh
conda activate hpc_env

### run .py file
srun python train_ddp_test_synthetic.py