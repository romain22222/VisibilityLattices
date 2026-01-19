#!/usr/bin/bash
BASE_COMMAND="./cmake-build-release/visibilityLattices --gpuRun -p $1 -g"
echo "Base command: $BASE_COMMAND"

for i in $(seq 1 -0.01 0.03); do
  tmp=$(printf "%.0f" $(echo "$i * 100" | bc))
#  echo "$BASE_COMMAND $i --computeNormals --computeCurvatures --nstar 3"
  $BASE_COMMAND $i --computeNormals --computeCurvatures --nstar 3
done