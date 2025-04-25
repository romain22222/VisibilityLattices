#!/bin/bash

BASE_COMMAND="$1 --noInterface -r $2 -p $3 -g"

for count in $(LANG=fr_FR seq 0.25 0.125 1); do
  $BASE_COMMAND $count
done