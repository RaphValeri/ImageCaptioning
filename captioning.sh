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
#SBATCH -J gpu-res
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
echo "FINE TUNNING HYPERPARAMETERS" >> $RESULTS_DIR/test.output
echo "3 cross attention layer at last Transformer block" >> $RESULTS_DIR/test.output
#echo "Added a gate to use or not the cross-attention" >> $RESULTS_DIR/test.output
echo "Training on 80% of the training data and validation on 20%" >> $RESULTS_DIR/test.output
echo "Only lower letters in the captions" >> $RESULTS_DIR/test.output
echo "Batch size of 16" >> $RESULTS_DIR/test.output
echo "----------------------------------" >> $RESULTS_DIR/test.output

TEMP=0.0
for N in 1 2
  do
    for EP in 1 2
      do
        torchrun --nproc_per_node 1 captioning_training.py --epochs $EP --nb_ca $N --loss_save_path res_files/CA_v2/${N}ca_ep${EP}/loss_${N}ca_ep${EP}.npy --model_path params_${N}ca_ep${EP}_v2.pt  >> $RESULTS_DIR/test.output
        torchrun --nproc_per_node 1 captioning_inference.py --model_path params_${N}ca_ep${EP}_v2.pt --nb_ca $N --p_test 0.1 --temperature $TEMP --json_path res_files/CA_v2/${N}ca_ep${EP}/eval_${N}ca_ep${EP}_t${TEMP}.json >> $RESULTS_DIR/test.output
      done
  done


