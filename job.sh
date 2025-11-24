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

# Do a for loop that prints the numbers from 1 to 0.01 in steps of 0.01
for i in $(seq 1 -0.01 0.01); do
  tmp=$(printf "%.0f" $(echo "$i * 100" | bc))
  /home/negror/VisibilityLattices/build/visibilityLattices -p tetrahedron -g $i --gpuRun --computeNormals --computeCurvatures --save "visibility-${tmp}.vis"
#  ./cmake-build-release/visibilityLattices -p tetrahedron -g $i --gpuRun --computeNormals --computeCurvatures --save "visibility-${tmp}.vis"
#  ./cmake-build-release/visibilityLattices -l
done
