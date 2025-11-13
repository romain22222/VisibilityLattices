#!/bin/bash
#SBATCH --job-name=visibilityTest
#SBATCH --partition=dayCPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB
#SBATCH --time=00:02:00

# export CUDA_VISIBLE_DEVICES=0
# echo $CUDA_VISIBLE_DEVICES

#export OMP_NUM_THREADS=128
echo $OMP_NUM_THREADS

/home/negror/VisibilityLattices/build/visibilityLattices -i /home/negror/VolGallery/Stanford-bunny/bunny-256.vol --gpuRun --computeNormals --computeCurvatures --save "visibility.vis" -r 20 -s 10