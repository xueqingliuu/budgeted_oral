#!/bin/bash
#SBATCH -J b10knownprob # Job name
#SBATCH -p murphy # Partition(s) (separate with
# commas if using multiple)
#SBATCH -c 12 # Number of cores
#SBATCH -t 0-10:00:00 # Time (D-HH:MM:SS)
#SBATCH --mem=50GB # Memory
#SBATCH --cpus-per-task=12
#SBATCH -o py_%j.o # Name of standard output file
#SBATCH -e py_%j.e # Name of standard error file
# load software environment
module load python
mamba activate budgeted
# print a statement
echo "This is the b10knownprob script"
# execute python code
python syn_exp_regret_known_prob.py

