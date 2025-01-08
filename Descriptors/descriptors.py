'''
Python program to extract SASA, DSSP, RMSD, RMSF from MDTraj 
'''
# Packages loading
import mdtraj as md
import os
import glob
import parmed as pmd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import math
from scipy.integrate import quad
import json
import pickle
import argparse
import matplotlib.pyplot as plt

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze molecular dynamics trajectory.')
    parser.add_argument('--trajectory', type=str, required=True, help='Trajectory file name')
    parser.add_argument('--reference', type=str, required=True, help='Reference file name')
    parser.add_argument('--base_name', type=str, required=True, help='Base name for output files or directories')
    args = parser.parse_args()

    # Load the trajectory and reference file
    fit_trj = md.load(args.trajectory, top=args.reference)
    reference = md.load(args.reference)

    # Solvent Accessible Surface Area (SASA) of just atoms within the protein
    prot_atoms=fit_trj.topology.select('protein')
    sliced_traj= fit_trj.atom_slice(prot_atoms)
    sasa_res = md.shrake_rupley(sliced_traj, mode='residue')
    sasa_at = md.shrake_rupley(sliced_traj, mode='atom')
    # Secondary structure (DSSP) analysis
    dssp_simple = md.compute_dssp(sliced_traj)
    dssp_comp = md.compute_dssp(sliced_traj,simplified=False)


    # RMSD calculation for alpha carbons
    atoms_ca = fit_trj.topology.select('name CA')

    rmsd = md.rmsd(fit_trj, reference, atom_indices=atoms_ca)
    
    # RMSF calculation for alpha carbons
    rmsf = md.rmsf(fit_trj, reference, 0, atom_indices=atoms_ca)

    time = fit_trj.time

    data = {
	'time': time,
	'sasa_res': sasa_res,
    'sasa_at':sasa_at,
    'dssp_simple': dssp_simple,
    'dssp_comp' : dssp_comp,
    'rmsd': rmsd,
    'rmsf': rmsf
    }

    with open(f'{args.base_name}_descriptors_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'{args.base_name}_sasa_res.pkl', 'wb') as f:
        pickle.dump(sasa_res, f)
    with open(f'{args.base_name}_sasa_at.pkl', 'wb') as f:
        pickle.dump(sasa_at, f)
    with open(f'{args.base_name}_dssp_simple.pkl', 'wb') as f:
        pickle.dump(dssp_simple, f)
    with open(f'{args.base_name}_dssp_complex.pkl', 'wb') as f:
        pickle.dump(dssp_comp, f)
    with open(f'{args.base_name}_rmsd.pkl', 'wb') as f:
        pickle.dump(rmsd, f)
    with open(f'{args.base_name}_rmsf.pkl', 'wb') as f:
        pickle.dump(rmsf, f)
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 15))

    
    # Plot SASA
    axs[0,0].plot(time, np.sum(sasa_res, axis=1))
    axs[0,0].set_title('Total SASA Over Time')
    axs[0,0].set_xlabel('Time (ps)')
    axs[0,0].set_ylabel('SASA (nm^2)')
    
    # Plot DSSP
    dssp_map={'H':0,'E':1,'C':2}
    dssp_int=np.vectorize(dssp_map.get)(dssp_simple)
    cmap = plt.cm.get_cmap('viridis',len(dssp_map))
    cax=axs[0,1].imshow(dssp_int.T, aspect='auto', cmap=cmap, interpolation='nearest')
    fig.colorbar(cax,ax=axs[0,1],ticks=range(len(dssp_map)),label="Secondary Structure")
    cax.set_clim(-0.5,len(dssp_map)-0.5)
    axs[0,1].set_title('DSSP simplified Over Time')
    axs[0,1].set_xlabel('Time (ps)')
    axs[0,1].set_ylabel('Residue')
    
    # Plot RMSD
    axs[1,0].plot(time, rmsd)
    axs[1,0].set_title('RMSD of Alpha Carbons Over Time')
    axs[1,0].set_xlabel('Time (ps)')
    axs[1,0].set_ylabel('RMSD (nm)')
    
    # Plot RMSF
    axs[1,1].plot(rmsf)
    axs[1,1].set_title('RMSF of Alpha Carbons')
    axs[1,1].set_xlabel('Residue Index')
    axs[1,1].set_ylabel('RMSF (nm)')
    
    plt.tight_layout()
    plt.savefig('analysis_plots.png')
    # plt.show()  # This line is commented out

if __name__ == '__main__':
    main()
