#!/bin/bash
#SBATCH --job-name=visibilityTest
#SBATCH --partition=dayCPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
## SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=00:10:00

# export CUDA_VISIBLE_DEVICES=0
# echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS=128
echo $OMP_NUM_THREADS

/home/negror/VisibilityLattices/build/visibilityLattices -i /home/negror/VolGallery/Stanford-bunny/bunny-1024.vol --noInterface -r 60