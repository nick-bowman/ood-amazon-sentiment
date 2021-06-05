#!/bin/bash
#SBATCH --exclude=jagupard[28-29]
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --mail-user=rmjones@cs.stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --requeue
#SBATCH --partition=jag-standard

set -x

cd $PWD
source /sailhome/rmjones/CS224U/.env/bin/activate
eval $1
