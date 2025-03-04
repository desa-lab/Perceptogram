#!/bin/sh
for sub in 1
do
    echo "Plotting ablation for subject $sub"
    # python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-clipvision --no-cliptext
    python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-vdvae --no-clipvision
    python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-clipvision
    python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-vdvae --no-cliptext
    python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-cliptext
    python thingseeg2_scripts/plot_reconstructions.py -sub $sub --no-vdvae
done