#!/bin/sh
for sub in 1 2 3 4
do
  for duration in 0 5 10 20 40 60 80
    do
        echo "sub $sub duration $duration"
        python thingseeg2_scripts/train_regression.py -sub $sub -duration $duration
    done
done

