#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/rv2018/files/MScProject/llama
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J captioning_inference
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
echo "----------------------------------" >> $RESULTS_DIR/test.output
echo "INFERENCE" >> $RESULTS_DIR/test.output
echo "Temperature = 0, 0.2, 0.4" >> $RESULTS_DIR/test.output
echo "----------------------------------" >> $RESULTS_DIR/test.output


torchrun --nproc_per_node 1 captioning_inference.py --model_path model_params_ca_ep1.pt --p_test 0.1 --temperature 0.1 --json_path eval_ca_ep1_t01.json >> $RESULTS_DIR/test.output

torchrun --nproc_per_node 1 captioning_inference.py --model_path model_params_ca_ep1.pt --p_test 0.1 --temperature 0.2 --json_path eval_ca_ep1_t02.json >> $RESULTS_DIR/test.output

torchrun --nproc_per_node 1 captioning_inference.py --model_path model_params_ca_ep1.pt --p_test 0.1 --temperature 0.3 --json_path eval_ca_ep1_t03.json >> $RESULTS_DIR/test.output

torchrun --nproc_per_node 1 captioning_inference.py --model_path model_params_ca_ep1.pt --p_test 0.1 --temperature 0.4 --json_path eval_ca_ep1_t04.json >> $RESULTS_DIR/test.output

torchrun --nproc_per_node 1 captioning_inference.py --model_path model_params_ca_ep1.pt --p_test 0.1 --temperature 0.5 --json_path eval_ca_ep1_t05.json >> $RESULTS_DIR/test.output