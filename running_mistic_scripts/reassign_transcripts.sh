#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=m05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G                    # memory per node
#SBATCH --time=1:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=/home/jholz/mistic_bayesian/reassign_logs/slurm-%j.out        # output file (%j = job ID) to capture logs for debugging
source /home/jholz/.bashrc
conda activate mistic2
echo "mistic_original"
echo "arg1 $1"
python /home/jholz/mistic_bayesian/MisTic-ablations/run_transcript_reassign.py -t $1 
