#!/bin/sh
for sub in 1 2 3 4
do
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 0 -gpu2 0 -duration 0 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 1 -gpu2 1 -duration 5 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 3 -gpu2 3 -duration 10 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 4 -gpu2 4 -duration 20 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 5 -gpu2 5 -duration 40 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 6 -gpu2 6 -duration 60 &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 7 -gpu2 7 -duration 80 &
    wait
done