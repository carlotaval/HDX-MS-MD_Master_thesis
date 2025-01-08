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
import myfunctions_frames as func
import argparse
import re
from collections import OrderedDict,defaultdict
'''
Calculations of hydrogen bonds scores, geometric, geometric with gaussian and electrostatic for each frame of the trayectory
Input:
*trajectory: .xtc file name with trajectory 
*ref_top: Reference topology necessary since .xtc does not keep that info, .gro works
*topology: topol.top files from gromacs, to be read with parmed
*base_name: base_name of the file used ie: barnase, secA, sec_A_ADP etc.
*frame: number of the frame to be proccessed
'''
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze molecular dynamics trajectory.')
    parser.add_argument('--trajectory', type=str, required=True, help='Trajectory file name')
    parser.add_argument('--ref_top', type=str,required=True, help='Reference topolofy necessary for xtc')
    parser.add_argument('--topology', type=str, required=True, help='Topology file name')
    parser.add_argument('--base_name', type=str, required=True, help='Base name for output files or directories')
    parser.add_argument('--frame', type=int, required=True, help='Frame to be analyzed')

    args = parser.parse_args()
    # Load the trajectory and reference file
    fit_trj = md.load_frame(args.trajectory,args.frame,top=args.ref_top)
    # Load the topology file
    topology = pmd.load_file(args.topology)
    
 
    # Extract atom information from topology to get charges later
    topology_data = []
    for atom in topology.atoms:
        topology_data.append({
            'index': atom.idx,
            'name': atom.name,
            'type': atom.type,
            'residue': atom.residue.name,
            'residue_index': atom.residue.idx,
            'charge': atom.charge,
            'mass': atom.mass
        })
    top_df = pd.DataFrame(topology_data)
    ################################################################
    # Iterate through each frame in the trajectory to create a dataframe with all the atom info and the coordenates
    for i, frame in enumerate(fit_trj.xyz):
        data = {
            'atom': [atom.name for atom in fit_trj.topology.atoms],
            'residue': [atom.residue.name for atom in fit_trj.topology.atoms],
            'residue_index': [atom.residue.index for atom in fit_trj.topology.atoms],
            'x': frame[:, 0],
            'y': frame[:, 1],
            'z': frame[:, 2],
            'time': fit_trj.time[i]
        }
    df = pd.DataFrame(data)
    # Topology file gets saved
    top_df.to_csv('topology_hbonds.csv', index=False) 

    #####################
    # Obtention of the possible atoms that form a Hydrogen bond 
    # This possible donors or acceptors atoms are always the same for all the trayectory so we can calculate the indices  just once
    Ac_Neigh_indices = {key: [] for key in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2', 'OG', 'OG1', 'OH', 'O']}    
    N_index = []
    H_index = []
    water_neigh_indices = []
    grouped = df.groupby('residue_index')
    for residue_index, residue_df in grouped:
            # Find the indices where the atom is 'N' and 'H' within the residue
        n_indices = residue_df.index[residue_df['atom'] == 'N'].tolist()
        h_indices = residue_df.index[residue_df['atom'] == 'H'].tolist()
            # Append non-empty lists to the total lists.
            # This is done so that N and H exist for each residue, since the NH part of the hbond are atoms from the same residue for one bond
            # Data from the N and H atoms is needed for the calculations
        if n_indices and h_indices:
            N_index.extend(n_indices)
            H_index.extend(h_indices)
        # Checks for acceptor atoms within the residue
        for acceptor in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2', 'OG', 'OG1', 'OH', 'O']:
                #Saves index and coords of acceptor atom
            acceptor_index = residue_df.index[residue_df['atom'] == acceptor].tolist()
            neighbour_atom = {
                                'OD1': 'CG', 'OD2': 'CG', 'OE1': 'CD', 'OE2': 'CD',
                                'ND1': 'CG', 'NE2': 'CE1', 'OG': 'CB', 'OG1': 'CB',
                                'OH': 'CZ', 'O': 'C'
                    }.get(acceptor, None) #If there is an acceptor atom is found, it saves the name of the name of the corresponding neighbour for each residue

            if neighbour_atom: #If it exists, it saves the index and the coords of the neighbour atom
                neighbour_index = residue_df.index[residue_df['atom'] == neighbour_atom].tolist()                    
                            #Checks that for each residue there is a acceptor and neighbour index and saves them
                if acceptor_index and neighbour_index:
                    index_ac_X = (acceptor_index,neighbour_index)
                                # Save the indices in the dictionary with the residue index as the key
                    Ac_Neigh_indices[acceptor].append(index_ac_X)
            # Before it saved O acceptors but now it also checks if they are waters and saves them apart
            if acceptor_index:
                if acceptor == 'O':
                    if 'HOH' in residue_df['residue'].values:
                        o_indices = residue_df.index[residue_df['atom'] == 'O'].tolist()[0]
                        h1_indices = residue_df.index[residue_df['atom'] == 'H1'].tolist()[0]
                        h2_indices = residue_df.index[residue_df['atom'] == 'H2'].tolist()[0]
                        if o_indices and h1_indices and h2_indices:
                            hydrogens=(h1_indices, h2_indices)
                            index_HOH = (o_indices, hydrogens)
                            water_neigh_indices.append(index_HOH)
    

        # N_index and H_index are lists with all the atoms indices of the protein
        # Ac_Neigh_indices saves the indices of all the possible acceptors and the corresponding neighbour
        # We want this indices in separate objects for acceptor Ac, and for the neighbour atom, X (atom connected to the Ac)
    Ac_indices_ls = []
    X_indice_ls = []
    water_indices = []
    water_h_indices = []
    for key in Ac_Neigh_indices.keys():
        for i in range(len(Ac_Neigh_indices[key])):
            Ac_idx,X_idx = Ac_Neigh_indices[key][i]
            Ac_indices_ls.append(Ac_idx)
            X_indice_ls.append(X_idx)
    Ac_index = []
    X_index = []
    for sublist in Ac_indices_ls:
        for item in sublist:
            Ac_index.append(item)
    for sublist in X_indice_ls:
        for item in sublist:
            X_index.append(item)

    for i in range(len(water_neigh_indices)):
            wat_O_idx, wat_H_idx =(water_neigh_indices[i])
            water_indices.append(wat_O_idx)
            water_h_indices.append(wat_H_idx)
    

    # Extraction of the coordenates for each frame
    coords_H,coords_N,coords_Ac,coords_X,coords_wat_0,coords_wat_H1,coords_wat_H2= func.extract_coordenates_water(df,H_index,N_index,Ac_index,X_index,water_indices,water_h_indices)

#######################
    # Calulations of distances, vectors and angles necessary 

    # distance_NH_vals = func.istance_element(coords_H,coords_N)
    #distance_AcX_vals = func.distance_element(coords_Ac,coords_X)

    vector_NH= func.vector_element(coords_H,coords_N)
    vector_AcX = func.vector_element(coords_X,coords_Ac)
    vector_HAc = func.vector_mx(coords_H,coords_Ac)
    vector_OH1 = func.vector_element(coords_wat_H1,coords_wat_0)
    vector_OH2 = func.vector_element(coords_wat_H2,coords_wat_0)
    vector_HO = func.vector_mx(coords_H,coords_wat_0)

    distance_HAc = func.distance_mx(coords_H,coords_Ac)
    distance_NAc_vals= func.distance_mx(coords_N,coords_Ac)
    distance_HO = func.distance_mx(coords_H,coords_wat_0)
    distance_NO = func.distance_mx(coords_N,coords_wat_0)

    alpha_angle= func.angle_alpha(vector_NH,vector_HAc)
    beta_angle = func.angle_beta(vector_AcX,vector_HAc)
    alpha_angle_wat= func.angle_alpha(vector_NH,vector_HO)
    beta_angle_watH1 = func.angle_beta(vector_OH1,vector_HO)
    beta_angle_watH2 = func.angle_beta(vector_OH2,vector_HO)


    # Indices of the matrix distance where the hydrogen bond exists
    index_hbond_water = func.indices_hbond(distance_HO,0.36)
    index_hbond = func.indices_hbond(distance_HAc,0.36) 

    # Calculations of scores for the protein-protein bonds and the protein-water bonds
    ### BONDS BETWEEN PROTEIN
    # Geometric hbonds
    geometric_H_bonds =[]
    for i in range(len(index_hbond)):
        row,col = index_hbond[i] 
        distance,alpha,beta = func.get_dis_angles(distance_HAc,alpha_angle,row,col,beta_angle1=beta_angle,beta_angle2=False) #Obatian distance and angles for each bond
        score = func.chemscore(distance,alpha,beta)
        pair=(row,col)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=Ac_index) #Get real atom indices instead of matrix indices
        bond = (H_atom,Ac_atom,score,'prot') #Saves real indices with the score
        geometric_H_bonds.append(bond)

    # Geometric hbonds gaussian
    geometric_H_bonds_gauss =[]
    for i in range(len(index_hbond)):
        row,col = index_hbond[i] 
        distance,alpha,beta = func.get_dis_angles(distance_HAc,alpha_angle,row,col,beta_angle1=beta_angle,beta_angle2=False) #Obatian distance and angles for each bond
        score = func.chemscore(distance,alpha,beta,gaussian=True)
        pair=(row,col)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=Ac_index) #Get real atom indices instead of matrix indices
        bond = (H_atom,Ac_atom,score,'prot') #Saves real indices with the score
        geometric_H_bonds_gauss.append(bond)
        
    # Electrostatic
    electrostatic_hbonds =[]
    for i in range(len(index_hbond)):
        H_mx_idx,Ac_mx_idx = index_hbond[i] #Get indices of the distance matrix 
        pair=(H_mx_idx,Ac_mx_idx)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=Ac_index) #Get real atom indices instead of matrix indices
        N_atom = N_index[H_mx_idx]#Get the N atom index of the corresponding N of the Hbond  to get the charge
        E = func.electrostatic(N_atom,H_mx_idx,Ac_mx_idx,H_atom,Ac_atom,top_df,distance_HAc=distance_HAc,distance_NAc_vals=distance_NAc_vals)
        result = (H_atom,Ac_atom,E,'prot')
        electrostatic_hbonds.append(result)
 

    ### BONDS BETWEEN PROTEIN AND SOLVENT
    # Geometric hbonds
    geometric_H_bonds_wat =[]
    for i in range(len(index_hbond_water)):
        row,col = index_hbond_water[i] #Get indices of the distance matrix 
        distance,alpha,beta1,beta2 = func.get_dis_angles(distance_HO,alpha_angle_wat,row,col,beta_angle_watH1,beta_angle_watH2) #Obtains distance and angles for each bond
        score = func.chemscore_water(distance,alpha,beta1,beta2) #Calculate the score
        pair=(row,col)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=water_indices) #Get real atom indices instead of matrix indices
        bond = (H_atom,Ac_atom,score,'wat') #Saves real indices with the score
        geometric_H_bonds_wat.append(bond)
        
    # Geometric hbonds
    geometric_H_bonds_gauss_wat =[]
    for i in range(len(index_hbond_water)):
        row,col = index_hbond_water[i] #Get indices of the distance matrix 
        distance,alpha,beta1,beta2 = func.get_dis_angles(distance_HO,alpha_angle_wat,row,col,beta_angle_watH1,beta_angle_watH2) #Obatian distance and angles for each bond
        score = func.chemscore_water(distance,alpha,beta1,beta2,gaussian=True) #Calculate the score
        pair=(row,col)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=water_indices) #Get real atom indices instead of matrix indices
        bond = (H_atom,Ac_atom,score,'wat') #Saves real indices with the score
        geometric_H_bonds_gauss_wat.append(bond)

    # Electrostatic hbonds
    electrostatic_hbonds_wat =[]
    for i in range(len(index_hbond_water)):
        H_mx_idx,Ac_mx_idx = index_hbond_water[i] #Get indices of the distance matrix 
        pair=(H_mx_idx,Ac_mx_idx)
        H_atom,Ac_atom = func.get_atoms(pair,indices_H=H_index,indices_Ac=water_indices) #Get real atom indices instead of matrix indices
        N_atom = N_index[H_mx_idx]#Get the N atom index of the corresponding N of the Hbond  to get the charge
        E = func.electrostatic(N_atom,H_mx_idx,Ac_mx_idx,H_atom,Ac_atom,top_df,distance_HAc=distance_HO,distance_NAc_vals=distance_NO)
        result = (H_atom,Ac_atom,E,'wat')
        electrostatic_hbonds_wat.append(result)
        
    # Bonds saved to dictionary.
    # Since this is for each frame, bonds is a dictionary with lists for each type of hydrogen-bond calculated  with 
    # Hydrogen atom index, Aceptor atom index the score and the type: prot (protein) or wat (water)
    bonds = defaultdict(list)
    bonds['electrostatic'].append(electrostatic_hbonds)
    bonds['electrostatic'].append(electrostatic_hbonds_wat)
    bonds['geometric'].append(geometric_H_bonds)
    bonds['geometric'].append(geometric_H_bonds_wat)
    bonds['geometric_gauss'].append(geometric_H_bonds_gauss)
    bonds['geometric_gauss'].append(geometric_H_bonds_gauss_wat)
    output_file = f'{args.base_name}_bonds_frame_{args.frame}.pkl'
    for key in bonds:
        bonds[key] = [item for sublist in bonds[key] for item in sublist]
        
    with open(output_file, 'wb') as f:
        pickle.dump(bonds, f)
        
if __name__ == '__main__':
    main()
