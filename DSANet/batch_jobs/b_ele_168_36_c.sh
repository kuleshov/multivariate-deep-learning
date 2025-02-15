#!/bin/bash
#SBATCH --job-name=dsae168_36c
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --output=jo_ele_168_36_c.txt
#SBATCH --error=je_ele_168_36_c.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list


echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Start running ... "
srun python3 single_gpu_trainer.py --data_name electricity --window 168 --horizon 36 --calendar True --batch_size 32 --split_train 0.7004694835680751 --split_validation 0.14929577464788732 --split_test 0.15023474178403756
echo "Finished running!"

seff $SLURM_JOBID

