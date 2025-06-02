# A complete tutorial to run a distributed training on multiple nodes with Pytorch DDP 

This repository documents the end-to-end steps to run a distributed training on multiple nodes with Pytorch DDP (Distributed Data Parallel) from scratch. In general, we have two major steps for this workload:
* Install and setup Slurm
* Run Pytorch DDP on Slurm

To install Slurm, we can follow the tutorial in this folder: [slurm_ubuntu_gpu_cluster](slurm_ubuntu_gpu_cluster)

In this repository, we have two examples on running Pytorch DDP on Slurm:
* Using CIFAR10 dataset: [cifar10_example](cifar10_example)
* Using synthetic dataset: [synthetic_data_example](synthetic_data_example)

Before we can run these examples, we need to first install conda. We can follow this [guideline](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install conda. Then, create a new conda environment and install required dependencies as follows:
```
conda create --name [YOUR_NEW_ENV_NAME] python=3.9
conda activate [YOUR_NEW_ENV_NAME]
pip install torch torchvision
```

To run the Pytorch DDP examples, we can run the following commands:
```
cd [PATH_TO_THE_EXAMPLE]
sbatch test.sh
```

For example, to run the example on CIFAR 10 dataset, we can run the following commands:
```
cd cifar10_example
sbatch test.sh
```

Before we can execute `sbatch test.sh`, we need to modify the `test.sh` file. In particular, we need to modify these lines:
* Edit the `--nodes` value at line 8 with the number of nodes in the Slurm cluster
* Edit the `--ntasks-per-node` value at line 9 with the number of GPUs in each node
* Edit the `--gres=gpu:` value at line 10 with the number of GPUs in each node
* Edit the `--cpus-per-task` value at line 11 with the number of CPUs we want to assign for each task
* Edit the `--mem` value at line 12 with the total memory per node you want to assign for the tasks  
* Change `\[PATH_TO_THE_MAIN_FOLDER\]` at line 13 with the absolute path of the example folder
* Change `\[YOUR_CONDA_ENV\]` at line 35 with the conda environment that we have created before