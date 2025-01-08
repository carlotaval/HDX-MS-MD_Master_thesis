## Script to calculate descriptors SASA,DSSP, RMSD and RMSF
## To use within a queue
#!/bin/bash
source /home/****/.bashrc
## directory must be where topol, em and fit are
source /****/apps/anaconda/anaconda3-2019.10/etc/profile.d/conda.sh
conda activate /home/****/env/hdx2
dir="$1"
base_name="$2"
cd "/****"

cd "$dir"

# Defines a tmp directory checking avilability
wd=$( date '+%F_%H-%M-%S' )
path=/PATH_JOBDIR/"$wd"
mkdir -p "$path"

# Copy necessary files
cp em.gro "$path"
cp topol* "$path"
cp fit_"$base_name".xtc "$path"
cp /****/scripts/descriptors.py "$path"

cd "$path"
"$CONDA_PREFIX"/bin/python3.9 descriptors.py --trajectory fit_"$base_name".xtc --reference em.gro --base_name "$base_name"


# Creation of local directory and copy of all the results
if [ ! -d "/****/$dir/descriptors_analysis" ]; then
  mkdir /****/"$dir"/descriptors_analysis
fi

cp analysis_plots.png /PATH TO/"$dir"/descriptors_analysis
cp *.pkl /****/"$dir"/descriptors_analysis

# Removal of temporal directory
cd /****/

rm -rf "$wd"
