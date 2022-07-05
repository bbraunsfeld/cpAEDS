#!/bin/sh
i=1
sbatch --ntasks 1 --cpus-per-task 4 --partition NGN --gres=mps:33 --mem=4096 $1