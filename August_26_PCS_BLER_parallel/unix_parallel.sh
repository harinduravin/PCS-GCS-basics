#!/bin/bash
#SBATCH --job-name="BER_plots"
#SBATCH --account=project_2013933
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

source /projappl/project_2013933/Sionna_New/bin/activate
python pn_multipath_ber_pcs_mp.py --workers 32 --ebno-min 8.0 --ebno-max 14.4 --ebno-step 0.2