#!/bin/bash

#$ -cwd
#$ -j y
#$ -pe smp 8        # Request cores (8 per GPU)
#$ -l h_vmem=7.5G   # Request RAM (7.5GB per core)
#$ -l h_rt=240:0:0    # Max 1hr runtime (can request up to 240hr)
#$ -l gpu=1         # Request GPU
#$ -N heaps5     # Name for the job (optional)

# Load the necessary modules
export OMP_NUM_THREADS=1
module load python/3.8.5
module load cudnn/8.1.1-cuda11


# Load the virtualenv containing the pytorch package
source ~/pvenv/bin/activate
# run the python script
python main.py
