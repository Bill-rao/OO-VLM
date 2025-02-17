#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH --partition=taodp_ffp
#SBATCH -N 1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


# load the environment
module purge
# source /public/software/profile.d/.bashrc
source /public/software/profile.d/apps_Anaconda3

conda activate /public/home/dapengtao_whr/.conda/envs/fusion

export PYTHONPATH=$PYTHONPATH:./slowfast
echo $PYTHONPATH
bash /public/home/dapengtao_whr/rao/fusion/exp/sthv2/ssv2_fusion_b/run.sh
#bash /public/home/dapengtao_whr/rao/fusion/exp/sthv2/ssv2_fusion_l_f32/run.sh


##
#1. Create a new conda environment
#2. Activate it
#3. conda install ***YOUR_DL_FRAMEWORK***  (The cuda version is 12.1)
#3. run ./launsh.sh
