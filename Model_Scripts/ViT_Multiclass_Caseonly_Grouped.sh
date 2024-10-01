#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -N ViT_Multiclass_Caseonly_Grouped

cd /rds/general/user/eg423/home/ML_Project_Group8/Eva
module load anaconda3/personal
source activate base

python ViT_Multiclass_Caseonly_Grouped.py