#!/bin/bash
# Master script to calculate hydrogen bonds for the molecular dynamics trajectories. 
# It is written to be used with a job scheduler like qsub
# It analyses each frame of the trajectory in paralell using the script 'hbonds_analysis_final.sh'
# Waits until this has been done for the 100001 frames to merge them together with 'merge_hbonds.py'
# It also returns 'topology_hbonds.csv' which is a dataframe of the topology 

dir=$1
base_name=$2
source /****/apps/anaconda/anaconda3-2019.10/etc/profile.d/conda.sh
conda activate ****** #activates directory 
cd  ***** #directory from when to call the next command, "dir" is been used as a relative path to this directory
#Calculations of hydrogen-bonds per frame
qsub -sync y -t 1:10001 -j y -o /dev/null -cwd -l mem_free=32G -l scratch_free=100G /PATH/TO/hbonds_analysis_final.sh "$dir" "$base_name"

input_dir="/PATH/TO/$dir/hbond_analysis" # this directory is created within hbonds_analysis_final.sh
output_dir="/PATH TO$dir" #output directory where you want to save the results of the merging

# Merge hydrogen bonds results
"$CONDA_PREFIX"/bin/python3.9 "/****/scripts/merge_hbonds.py" --directory "$input_dir" --base_name "$base_name" --outputdir "$output_dir"
cp "/****/$dir/hbond_analysis/topology_hbonds.csv" "$output_dir"

if [ -f "/PATH/TO/${dir}/${base_name}_bonds_prot.pkl" ]; then
  rm -rf "$input_dir"
fi
