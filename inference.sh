#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/rv2018/archive/ImageCaptioning
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J gpu-inference
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=3-20:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=20000
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#------------------------------
# e.g.:
# Load the conda environment
flight env activate conda
conda activate torchEnv


#===========================
#  Create results directory
#---------------------------
RESULTS_DIR="$(pwd)/${SLURM_JOB_NAME}-outputs/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

#===============================
#  Application launch commands
#-------------------------------
# Customize this section to suit your needs.

echo "Executing job commands, current working directory is $(pwd)"

# REPLACE THE FOLLOWING WITH YOUR APPLICATION COMMANDS

N=1
EP=1
for TEMP in 0.1 0.2 0.3 0.4 0.6 0.8 1.0
  do
    torchrun --nproc_per_node 1 captioning_inference.py --model_path params_${N}ca_ep${EP}.pt --nb_ca $N --p_test 0.1 --temperature $TEMP --json_path res_files/${N}ca_ep${EP}/eval_${N}ca_ep${EP}_t${TEMP}.json >> $RESULTS_DIR/test.output
  done