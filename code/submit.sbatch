#!/bin/bash
#
#SBATCH --job-name=bd  # Job name for tracking
#SBATCH --partition=gecko       # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job, 16 for gecko, 12 for flacon
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --time=2-00:00:00       # Job time limit set to 2 days (48 hours)
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

cd /dcs/large/u2157170/code/modelv2/

# python3.11 train.py
# python3.11 eval_selfcontained.py
# python3.11 distribution_check.py
# python3.11 bdcurves.py
# python3.11 resultsanalysisminnen.py
# python3.11 writeupbdcurvesballeALL.py
# python3.11 modelseval.py
python3.11 totalrdcurves.py