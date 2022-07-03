#!/bin/bash
#SBATCH -p htc
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH -p development
#SBATCH -o assignment_05_%A-%a.out
#SBATCH --array=1-2
#SBATCH --job-name=hpc_assignment_05_ricco_ferraro


# Note I did the same multi job script for assignment 05
cmd=$(cat <<-END
#import socket
#print(f"hostname: {socket.gethostname()}")
print("running python in signularity container")
END
)

singularity exec ./lab02.sif /opt/view/bin/python3.8 -c "$cmd"

echo "============= job id ============="
echo "jobID is: $SLURM_JOBID"

echo "============= node hostname =============="
hostname=$(srun hostname)
echo "$hostname"

echo "============= current node info ============="
scontrol show node "$hostname"

echo "============= all job info ============="
sinfo --long

echo "============= proc cpu info ============="
proc_info=$(cat /proc/cpuinfo)
echo "$proc_info"
echo "============= free info ============="
free -g