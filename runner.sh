#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=80:00:00

python cheng_Blasto_F.py   #Change the file name accordingly
