#!/bin/bash
# Script to send to the job scheduler the calcularion of hydrogen bonds
# Load the necessary environment
source /****/apps/anaconda/anaconda3-2019.10/etc/profile.d/conda.sh
source /PATH/TO .bashrc
conda activate /PATH/TO/env/hdx2
# Directory where results are
dir="$1"
base_name="$2"
#This is usually runned from HOME directory, in the cluster used the home directory was not the right one, hence why it is changed
cd "PATH" 
cd "$dir"
echo $PWD
# Define a temporary directory
wd=$(date '+%F_%H-%M-%S')
wd_final="$wd"_"$SGE_TASK_ID"
path=/PATH_JOBDIR/"$wd_final" # path to directory where execution happends within the job scheduler
task_id=$(($SGE_TASK_ID - 1))
mkdir -p "$path"

# Copy necessary files
cp fit_"$base_name".xtc "$path"
cp topol* "$path"
cp em.gro "$path"
cp posre* "$path"
cp -r ./lig_*.acpype "$path"

cd "$path"

echo $(ls)
export PATH=/lrlhps/apps/gromacs/gromacs-2020.4/bin:$PATH

# Calculate the time for the current frame (assuming 2 fs per step) 
echo $CONDA_PREFIX
# hbond_frames_final.py extracts the frame and performs hydrogen bond analysis
"$CONDA_PREFIX"/bin/python3.9 "/PATH TO/hbond_frames_final.py" --trajectory fit_"$base_name".xtc --ref_top em.gro --topology topol.top --base_name "$base_name" --frame "$task_id"

# Move the output files to a new directory
output_dir="/PATH/$dir/hbond_analysis/"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi
cp "$base_name"_bonds*.pkl "$output_dir"
cp topology_hbonds.csv "$output_dir"

# Remove temporary directory
cd /PATH_JOBDIR
rm -rf "$wd_final"


