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


for (( i = 1; i > 0; i-=0.01 )); do
  echo "Running visibility computation with parameter i=${i}"
#  /home/negror/VisibilityLattices/build/visibilityLattices -p tetrahedron -g i --gpuRun --computeNormals --computeCurvatures --save "visibility-${i}.vis" -r 20 -s 10
done
