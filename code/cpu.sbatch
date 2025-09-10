#!/bin/bash
#
#SBATCH --job-name=tf3 # Job name for tracking
#SBATCH --partition=cpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=1      # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem-per-cpu=1500     # RAM in MB needed per core (i.e. 1500MB * 40 threads = 60GB RAM)
#SBATCH --time=02:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
##source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
##module load CUDA
conda activate testenv

## echo "GPU CHECK" ##
## nvidia-smi || echo "no GPU" ##
## echo $CUDA_VISIBLE_DEVICES ##

# List only top-level entries in a .tar.zst archive
cd /dcs/large/u2157170/code/

# tar --use-compress-program=unzstd \
#     -xvf BigEarthNet-S2.tar.zst \
#     -C /dcs/large/u2157170/testfold4/ \
#     BigEarthNet-S2/S2B_MSIL2A_20170829T105019_N9999_R051_T31UER



python3.11 combinebandsall.py