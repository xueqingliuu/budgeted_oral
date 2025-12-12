#!/bin/bash
#SBATCH -J b20known10 # Job name
#SBATCH -p test # Partition(s) (separate with
# commas if using multiple)
#SBATCH -c 1 # Number of cores
#SBATCH -t 0-10:00:00 # Time (D-HH:MM:SS)
#SBATCH --mem=50GB # Memory
#SBATCH -o py_%j.o # Name of standard output
file
#SBATCH -e py_%j.e # Name of standard error file
# load software environment
module load python
manba activate budgeted
# print a statement
echo "This is the b20known10 script"
# execute python code
python syn_exp_regret_known_10.py

