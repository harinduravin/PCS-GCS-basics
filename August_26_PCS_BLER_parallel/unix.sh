#!/bin/bash
#SBATCH --job-name="BER_plots"
#SBATCH --account=project_2013933
#SBATCH --mem=17500
#SBATCH --time=1:00:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100_1g.5gb:1

source /projappl/project_2013933/Sionna_New/bin/activate
nvidia-smi
python pn_multipath_ber_pcs.py