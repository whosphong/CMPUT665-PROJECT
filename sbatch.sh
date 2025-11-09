#!/bin/bash -ex
#SBATCH --output=/home/qhp/project/deep_rl/slurm/slurm_%x.out   
#SBATCH --error=/home/qhp/project/deep_rl/slurm/slurm_%x.err   
#SBATCH --nodes=1   
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=46GB   
#SBATCH --cpus-per-gpu=16      
#SBATCH --mail-type=all   
#SBATCH --time=8:00:00
#SBATCH --dependency=1860701

module load cuda

cd /home/qhp/project/deep_rl  
source .venv/bin/activate

echo "Starting job $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Environment: $ENV_NAME"
echo "Algorithm: $ALGO_NAME"
echo "SEED: $SEED"
echo "Extra Args: $EXTRA_ARGS"

python run_mujoco_by_step.py --env "$ENV_NAME" --algo "$ALGO_NAME" --seed "$SEED" $EXTRA_ARGS