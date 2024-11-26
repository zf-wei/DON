#!/bin/bash
#SBATCH --job-name=DeepONet_Variants     # Job name
#SBATCH --partition=a100_shared          # Partition name (dlt)
#SBATCH --mail-type=ALL                  # Send email on all events
#SBATCH --mail-user=zfwei@pnnl.gov       # Your email for notifications
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --time=8:00:00                  # Time limit hrs:min:sec
#SBATCH -A ascr_dpdf                     # Account name (ascr_dpdf)
#SBATCH --array=0-4,6                      # Array range for parameter combinations


# Activate Conda environment
source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh
conda activate test

# Run the Python script with the current set of parameters
python 1d.py --problem "burgers" --var ${SLURM_ARRAY_TASK_ID} --struct 1 --sensor 101
