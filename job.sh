#!/bin/bash
#SBATCH --job-name=visibilityTest
#SBATCH --partition=dayGPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --time=00:10:00

echo $CUDA_VISIBLE_DEVICES
/home/negror/VisibilityLattices/build/visibilityLattices -i /home/negror/VolGallery/Stanford-bunny/bunny-64.vol --noInterface -r 20