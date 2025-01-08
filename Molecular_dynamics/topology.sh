#!/bin/bash
# Script to process the initial pdb structure with GROMACS
# Input: pdb structure cleand and without waters
# Last command expects a manual input choosing which group within the structre to substituted with ions
# It has been let as a manual input since GROMACS groups defined by gromacs change depending on the systems (ligands,ions etc)

# source /PATH/TO/GMXRC.bash # Modify path
input_pdb="$1"
base_name="${input_pdb%.pdb}"

# .pdb to .gmx format conversion
output_gmx="${base_name}_processed.gro"
( echo "6" ; echo "1" ) | gmx pdb2gmx -f "$input_pdb" -o "$output_gmx"

# Creation of boundary to fill with waters
output_newbox="${base_name}_newbox.gro"
gmx editconf -f "$output_gmx" -o "$output_newbox" -c -d 1.2 -bt cubic

#Solvation
output_solv="${base_name}_solv.gro"
gmx solvate -cp "$output_newbox"  -cs spc216.gro -o "$output_solv" -p topol.top

# Neutralization with ions
gmx grompp -f PATH/TO/PARAMS/ions.mdp -c "$output_solv" -p topol.top -o ions.tpr # Update params path
output_solv_ions="${base_name}_solv_ions.gro"
gmx genion -s ions.tpr -o "$output_solv_ions" -p topol.top -pname NA -nname CL -neutral 

