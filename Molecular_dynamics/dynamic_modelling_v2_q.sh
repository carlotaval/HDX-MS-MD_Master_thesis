#!/bin/bash
# Script to do energy minimization, equilibratioin and molecular dynamics
# It is supposed to be runned after topology.sh
# It uses GROMACS and it is designed to use with a job scheduler like qsub
# Path to executable of Gromacs must be changed
# Adjust threads (-nt ) when necessary


# Input 'files_dir' is the path to the directory where input files are from the HOME directory
# Running jobs with qsub set the PWD directory to HOME
files_dir="$1" 
#Input 'input_pd'b is the pdb file used previously with topology.sh 
input_pdb="$2"
# Input 'replicate' is an integer for the nº of replicate that will be generated (1,2,3...)
replicate="$3"

source PATH/TO/GMXRC.bash #Change path to where GMXRC.bash is

# Generation of new base_name
base="${input_pdb%.pdb}"
base_name="$base"_rep_"$replicate"

# Defines a tmp directory 
wd="${base_name}_$(date '+%F_%H-%M-%S')"
path=/PATH_TO_TMP/"$wd" #Update path where jobs are runned if necessary when using job scheduler
# Name of the input file
output_solv_ions="${base}_solv_ions.gro"

# Change to the files directory  and creation of the path directory
cd "$files_dir"
mkdir -p "$path"

# Copy necessary files
cp "$output_solv_ions" "$path"
cp topol* "$path"
cp posre* "$path"
cp -r "PATH/TO/params" "$path" #Update path to params folder
cp -r ./lig_*.acpype "$path"

#Changes directory to temporal to do computations
cd "$path"

# Energy Minimization
gmx grompp -f ./params/minim.mdp -c "$output_solv_ions" -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# Equlibration fixating temperature
gmx grompp -f ./params/nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt -nt 16
( echo "16" ) | gmx energy -f nvt.edr -o temperature.xvg

# Equlibration fixating Pressure
gmx grompp -f ./params/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -deffnm npt -nt 16
( echo "18" ) | gmx energy -f npt.edr -o pressure.xvg

# Molecular dynamics
gmx grompp -f ./params/md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_"$base_name"_0_100.tpr
gmx mdrun -deffnm md_"$base_name"_0_100 -nt 16

# Deletion of boundaries
( echo "0" )| gmx trjconv -s md_"$base_name"_0_100.tpr -f md_"$base_name"_0_100.xtc -o pbc_whole_"$base_name".xtc -pbc whole 
( echo "1" , echo "0" )| gmx trjconv -s md_"$base_name"_0_100.tpr -f pbc_whole_"$base_name".xtc -o pbc_molcenter_"$base_name".xtc -pbc mol -center

## FIT TO ONE POINT
( echo "0" , echo "0" ) | gmx trjconv -f  pbc_molcenter_"$base_name".xtc -s em.tpr -fit rot+trans -o fit_"$base_name".xtc

##########################################################################
# Creation of local directory and copy of all the results
# Define the directory path
DIR="/****/${files_dir}/results"

# Check if directory exists
if [ -d "$DIR" ]; then
  cp -r "$path" "$DIR"
else
  echo "Directory $DIR does not exist. Creating it now."
  mkdir -p "$DIR"
  cp -r "$path" "$DIR"
fi

# Removal of temporal directory
cd /PATH_TO_TMP/

rm -rf "$wd"

