#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1     # Request any available GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # There are 32 CPU cores on v100 Cedar GPU nodes
#SBATCH --mem=32G              # Request the full memory of the node
#SBATCH --time=80:00:00

python cheng_Blasto_F.py   #Change the file name accordingly
