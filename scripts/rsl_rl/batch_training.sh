#!/bin/bash

#SBATCH -N 1                      # Number of nodes requested
#SBATCH -n 1                      # Number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4         # Number of cores per task
#SBATCH --gres=gpu:l40s:1         # Request a L40S GPU
#SBATCH --nodelist=al-l40s-0.grasp.maas
#SBATCH --qos=al-high-2gpu        # QoS
#SBATCH --partition=aloque-compute
#SBATCH -t 1-00:00                # Maximum time (24h)
#SBATCH -D /home/lorebia/Github/isaac_quad_sim2real

##SBATCH -o slurm.%N.%j.out
##SBATCH -e slurm.%N.%j.err

# Print some info for context
pwd
hostname
date

echo "Starting job..."

source ~/.bashrc
conda activate isaaclab_45
export PYTHONPATH=$PYTHONPATH:/home/lorebia/Github/isaac_quad_sim2real/src

export PYTHONUNBUFFERED=1

# Sequentially launch
for seed in 1 2 3; do
    echo "Running with seed $seed..."
    python scripts/rsl_rl/train_race.py \
        --task Isaac-Quadcopter-Race-v0 \
        --num_envs $((12288*2)) \
        --max_iterations 5000 \
        --headless \
        --logger wandb \
        --seed $seed
    echo "Finished run with seed $seed"

    wait
    sleep 30
done

# Print completion time
date
