#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /tmp/budgeted_oral_results/eval_pooling/out_%j.txt # File to which STDOUT will be written
#SBATCH -e /tmp/budgeted_oral_results/eval_pooling/err_%j.txt # File to which STDERR will be written

python -u run.py /tmp/budgeted_oral_results/eval_pooling eval_pooling '{"cluster_size": "no_pooling", "offline_or_online": "online", "state": null, "seed": 1}'
