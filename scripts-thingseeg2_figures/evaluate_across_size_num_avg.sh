#!/bin/sh
for size in 500 1000 2000 4000 6000 10000 14000
# for size in 2000 4000 6000 10000 14000
do
  for avg in 5 10 20 30 40 60
  do
    python thingseeg2_scripts/evaluate_reconstruction.py -sub 1 -size $size -avg $avg
  done
  python thingseeg2_scripts/evaluate_reconstruction.py -sub 1 -size $size
done
for avg in 5 10 20 30 40 60
do
  python thingseeg2_scripts/evaluate_reconstruction.py -sub 1 -avg $avg
done
