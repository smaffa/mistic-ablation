#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=mistic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G                    # memory per node
#SBATCH --time=12:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=/home/jholz/mistic_bayesian/mistic_runs/slurm/slurm-%j.out        # output file (%j = job ID) to capture logs for debugging
source /home/jholz/.bashrc
echo "mistic_nospatial"
conda activate mistic2
echo "arg1 $1"
echo "arg2 $2"
mkdir $2
python /home/jholz/mistic_bayesian/MisTic-ablations/run_mistica_script.py -t $1 -o $2