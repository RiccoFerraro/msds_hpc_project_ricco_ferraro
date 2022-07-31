#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -p gpgpu-1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=32G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-02:00:00



# activate venv
source project_venv/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0
module load cuda

# run script from above
srun --pty -p gpgpu-1 -n 1 -c 36 --mem=32G --gres=gpu:1 python3 src/project_lightning.py --num_nodes=2 --num_devices=1
# nvidia-smi