# (submit.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -p standard
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=6G
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

# run script from above
srun python3 src/project_lightning.py --num_nodes=2 --num_devices=1