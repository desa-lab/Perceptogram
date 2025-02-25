#!/bin/sh
for sub in 1 2 3 4
do
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 0 -gpu2 0 --no-vdvae --no-clipvision &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 1 -gpu2 1 --no-clipvision &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 3 -gpu2 3 --no-vdvae --no-cliptext &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 4 -gpu2 4 --no-cliptext &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 5 -gpu2 5 --no-vdvae &
    python thingseeg2_scripts/reconstruct_from_embeddings.py -sub $sub -gpu1 6 -gpu2 6
    wait
done