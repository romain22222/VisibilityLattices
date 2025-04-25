#!/bin/bash

BASE_COMMAND="$1 --noInterface -p $2 -g"

for count in $(LANG=fr_FR seq 0.125 0.125 1); do
  echo "Gridstep: $count"
  $BASE_COMMAND $count
done