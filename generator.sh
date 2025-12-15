#!/usr/bin/bash
BASE_COMMAND="./cmake-build-release/visibilityLattices --gpuRun -p $1 -g"
echo "Base command: $BASE_COMMAND"

for i in $(seq 1 -0.01 0.02); do
  tmp=$(printf "%.0f" $(echo "$i * 100" | bc))
  $BASE_COMMAND $i --noFurtherComputation --saveShapeFilename "./shapes/shape-${tmp}.vol"
done