#!/bin/sh
for sub in 1 2 3 4
do
    echo "Evaluating ablation for subject $sub"
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-clipvision --no-cliptext
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-vdvae --no-clipvision
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-clipvision
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-vdvae --no-cliptext
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-cliptext
    python thingseeg2_scripts/evaluate_reconstruction.py -sub $sub --no-vdvae
done