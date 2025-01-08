#Functions used throught analysis
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict, OrderedDict
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.special import rel_entr
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.lines as mlines  
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from scipy.stats import iqr 
from sklearn.preprocessing import PolynomialFeatures


def median_absolute_deviation(data):
    median = np.median(data)
    deviation = np.abs(data - median)
    mad = np.median(deviation)
    return mad
def mean_absolute_deviation(data):
    mean = np.mean(data)
    deviation = np.abs(data - mean)
    mad = np.mean(deviation)
    return mad
        
def gibbs_proc(gibbs):
    '''Processes gibbs files since headers are not in need'''
    gibbs_proc=gibbs.copy()
    gibbs_proc = gibbs_proc.drop(([0,2]))
    gibbs_proc.columns = gibbs_proc.iloc[0]
    gibbs_proc = gibbs_proc[1:]
    gibbs_proc.loc[:, 'quantity'] = gibbs_proc['quantity'].astype(int)
    gibbs_proc.loc[:, 'dG'] = gibbs_proc['dG'].astype(float)
    return gibbs_proc
    
def hbonds_per_res(file_name,nt_correction,topology_file,dictionary,em=False):
    '''Takes hbonds files and calculates for each bond the strongest bond if more than one acceptors.
    Outout is a dictionary where keys are the atoms (atom index) that act as donors and the values are an array of the highest scores for each frame)'''
    with open(file_name,'rb') as file:
        rep=pickle.load(file)
    topology=pd.read_csv(topology_file)
    # Creation of empty dicitonary of dictionaries whith list as values
    values = defaultdict(lambda: defaultdict(lambda:np.zeros(10001)))
    for key in rep[dictionary].keys():
        for frame, acceptor, score in rep[dictionary][key]: 
            values[key][acceptor][frame]=score 
            #Values main keys are donor atoms number, then within each key, there are a key for each acceptor which has a list of 10001 where for each value the corresponding score to the frame is saved
    #Values all acceptors is a dicitionary to save as key the donors and then combined the info of all acceptors
    values_all_acceptors = defaultdict((lambda: np.zeros(10001)))
    names_aa=[]
    for key in values.keys():
        array=np.zeros(10001)
        for i in range(10001):
            acceptors = values[key].keys() #List of acceptors for each donor
            scores=[]
            for acceptor in acceptors:
                scores.append(values[key][acceptor][i])
            array[i]=max(scores) #Save for each frame the score of the highest acceptor if there is more than one
        res=topology.iloc[key]['residue_index']+nt_correction
        values_all_acceptors[res]=array
    if em==True:
        #for just em
        for key in values_all_acceptors.keys():
            values_all_acceptors[key]=values_all_acceptors[key][0]
    return values_all_acceptors

def unpivot_Duptake(D_uptake):
    '''Unpivots Deuterium uptake files to be able to take values for each specific time'''
    pivot_D_uptake = D_uptake.pivot_table(index=['start', 'end', 'sequence', 'state'],columns='exposure', values='uptake').reset_index()
    D_uptake_prot=pivot_D_uptake[pivot_D_uptake['state'] != 'Full deuteration control']
    D_uptake_prot.columns = D_uptake_prot.columns.map(str)
    return D_uptake_prot


def distribution_hbonds(geom, geom_gauss, elect, topology, nt_correction,base_name,replicate, save_dir=None):
    """
    Plots hydrogen bond distributions from bond data without subplot limits.

    Parameters:
    - geom (dict): Dictionary containing geometric score data for each residue.
    - geom_gauss (dict): Dictionary containing Gaussian geometric score data for each residue.
    - elect (dict): Dictionary containing electrostatic score data for each residue.
    - topology (pd.DataFrame): DataFrame with topology information.
    - nt_correction (int): Index correction for nucleotide positions.
    - save_dir (str): Directory to save the plots. If None, plots are only displayed.
    """


    # Parameters
    num_cols = 6  # Total columns for the grid
    num_rows = 6  # Total rows for the grid
    plots_per_residue = 3  # Number of plots per residue
    batch_size = num_cols * num_rows // plots_per_residue  # Residues per figure (6x6 -> 12 residues)

    # Loop over residues in batches
    for start_idx in range(0, len(geom), batch_size):
        end_idx = min(start_idx + batch_size, len(geom))  # Ensure we don't exceed available data

        # Create figure with 6x6 grid layout
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))  # Adjust size for compact layout
        axs = axs.flatten()  # Flatten for easier indexing

        plot_idx = 0  # Subplot index
        residue_keys = list(geom.keys())[start_idx:end_idx]  # Get residue keys for this batch

        for i in range(0, len(residue_keys), 2):  # Process residues in pairs (2 residues per row)
            if i + 1 >= len(residue_keys):  # If there's an odd number, handle the last residue separately
                keys = [residue_keys[i]]  # Only 1 residue left
            else:
                keys = [residue_keys[i], residue_keys[i + 1]]  # Process in pairs

            for j, key in enumerate(keys):  # Process each residue in the pair
                # Get residue name
                Res_name = topology[topology['residue_index'] == key - nt_correction]['residue'].values
                Res_name = Res_name[0] if len(Res_name) > 0 else f"Res_{key}"
                res_index = key-nt_correction
                # Plot 3 metrics for this residue
                axs[plot_idx].hist(geom[key], bins=20)
                axs[plot_idx].set_title(f'{Res_name} {res_index}', fontsize=16)
                axs[plot_idx].set_xlabel('Geométrica', fontsize=14)
                axs[plot_idx].set_ylabel('Freq', fontsize=14)

                axs[plot_idx + 1].hist(geom_gauss[key], bins=20)
                axs[plot_idx + 1].set_title(f'{Res_name} {res_index}', fontsize=16)
                axs[plot_idx + 1].set_xlabel('Gaussiana', fontsize=14)
                axs[plot_idx + 1].set_ylabel('Freq', fontsize=14)

                axs[plot_idx + 2].hist(elect[key], bins=20)
                axs[plot_idx + 2].set_title(f'{Res_name} {res_index}', fontsize=16)
                axs[plot_idx + 2].set_xlabel(' Electróstatica', fontsize=14)
                axs[plot_idx + 2].set_ylabel('Freq', fontsize=14)
                fig.suptitle(f'Distribución de las puntuaciones para  la proteína \n {base_name} réplica:{replicate}', y=1.02, fontsize=16)
                fig.subplots_adjust(top=0.85, hspace=0.5)  # Adjust top margin

                # Move to next set of 3 plots
                plot_idx += 3

        while plot_idx < len(axs):  
            fig.delaxes(axs[plot_idx])  # Remove unused subplot
            plot_idx += 1   
        # Adjust layout to avoid overlap
        plt.tight_layout(pad=0.5)

        # Save the figure

        if save_dir:
            plt.savefig(f'compact_residue_plots_{start_idx // batch_size + 1}.png', dpi=150)

        # Show the figure (optional, can be commented out if saving only)
        plt.show()

        # Close the figure to free memory
        plt.close(fig)



def data_extractionX(files_names):
    '''Takes as input the directory, base_name, nt_correction and ligand (nt_correction is because topologies, real HDX experiemnt do not use same indexing) in a list of lists
    It returns a dataframe with this columns:
    -id: id made of base_name, replicate, residue name and residue number
    -protein
    -replicate
    -name of residue
    -nº of residue
    -frame
    -score for bonds wiithin protein residues
    -score for bonds with water
    -value of sasa for the atom
    -value of sasa for the atom
    -value for helix, 1 if true 0 if not
    -value for beta,  1 if true 0 if not
    -value for coil, 1 if true 0 if not
    '''
    columns = ['id', 'prot', 'replicate', 'res_name', 'res_idx', 'frame', 'scores_prot','scores_water', 'sasa_at','sasa_res', 'helix', 'beta', 'coil']
    data_list=[]
    for directory, base_name, nt_correction,ligand in files_names:
        print('analysing: ',base_name)
        for j in range(3):
            j=j+1
            replicate=j
            # File names
            bonds_suffix = f'_rep_{replicate}_bonds_prot.pkl'
            water_bonds_suffix = f'_rep_{replicate}_bonds_water.pkl'
            bonds_name = base_name + bonds_suffix 
            water_bonds_name = base_name + water_bonds_suffix
            descriptors_name = f'{base_name}_rep_{replicate}_descriptors_data.pkl'
            bonds_dir = f"/****sim/{directory}/rep{replicate}"
            descriptors_dir = f"/****/sim/{directory}/rep{replicate}/descriptors_analysis"
            prot_file = bonds_dir + '/' + bonds_name
            water_file = bonds_dir + '/' + water_bonds_name
            descriptors_file = descriptors_dir + '/' + descriptors_name
            topology_file =bonds_dir+'/topology_hbonds.csv'
            # Data load
            with open(descriptors_file, 'rb') as file:
                data_descriptors = pickle.load(file)
            top = pd.read_csv(topology_file)
            scores_prot = hbonds_per_res(prot_file, nt_correction, topology_file, 'geometric')
            scores_water = hbonds_per_res(water_file, nt_correction, topology_file, 'geometric')
                
            sasa_values = data_descriptors['sasa_at']
            H_indices = top.index[top['name'] == 'H'].tolist()
            if ligand != None:
                H_indices = [i for i in H_indices if top.loc[i, 'residue'] != ligand]

            sasa_mapping = {top.iloc[idx]['residue_index'] + nt_correction: data_descriptors['sasa_at'][:, idx] for idx in H_indices}
            sasa_res=data_descriptors['sasa_res']
            sasa_res_mapping = {i+nt_correction: sasa_res[:, i] for i in range(sasa_res.shape[1])}
            dssp_values = data_descriptors['dssp_simple']
            one_hot_encoded = np.zeros((dssp_values.shape[0], dssp_values.shape[1], 3), dtype=int)
            for i in range(dssp_values.shape[0]):
                for k in range(dssp_values.shape[1]):
                    if dssp_values[i, k] == 'H':
                        one_hot_encoded[i, k] = [1, 0, 0]
                    elif dssp_values[i, k] == 'E':
                        one_hot_encoded[i, k] = [0, 1, 0]
                    elif dssp_values[i, k] == 'C':
                        one_hot_encoded[i, k] = [0, 0, 1]
            dssp_mapping = {i + nt_correction: one_hot_encoded[:, i] for i in range(dssp_values.shape[1])}
            top['residue_index'] = top['residue_index'] + nt_correction
            mean = np.mean(data_descriptors['rmsd'])
            index = next((i for i, value in enumerate(data_descriptors['rmsd']) if value > 0.35), 0)

            # Keep values after index
            for key in scores_prot.keys():
                if key in sasa_mapping.keys() and key in dssp_mapping.keys():
                    res_name = top[top['residue_index'] == key].iloc[0,:]['residue']
                   # if mean <0.35:
                    #    val=10001
                    #else:
                     #   val=index
                    for f in range(0,index):
                        row_values ={
                            'id': f'{base_name}_{replicate}_{res_name}{key}',
                            'prot': base_name,
                            'replicate': replicate,
                            'res_name': res_name,
                            'res_idx': key,
                            'frame': f,
                            'scores_prot': scores_prot[key][f],
                            'scores_water': scores_water[key][f],
                            'sasa_at':  sasa_mapping[key][f],
                            'sasa_res': sasa_res_mapping[key][f],
                            'helix': dssp_mapping[key][f][0],
                            'beta': dssp_mapping[key][f][1],
                            'coil': dssp_mapping[key][f][2]
                        }
                        data_list.append(row_values)
    data = pd.DataFrame(data_list, columns=columns)
    return data
    
def data_extractionY(files_names):
    '''
    Function to extract values for gibbs results, takes as input the base name, since all files take same format
    Output is a dataframe with thsi columns:
    id= base_name, replicate, residue name and number
    protein
    replicate
    residue name
    residue number
    gibbs energy
    '''
    columns = ['id', 'prot', 'replicate', 'res_name', 'res_idx', 'dG']
    data_list=[]
    for base_name in files_names:
        for j in range(3):
            j=j+1
            replicate=j
            # File names
            gibbs_name=base_name[0]+'_gibbs.csv'
            gibbs_dir=f'/****/sim/gibbs_tables'
            gibbs_file=gibbs_dir+'/'+gibbs_name    
            # Data load
            gibbs=gibbs_proc(pd.read_csv(gibbs_file,header=None))
            dict_aa = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
             'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 
             'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 
                 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET'}
            for index, row in gibbs.iterrows():
                res_idx = row['quantity']
                sequence = row['sequence']
                dG = float(row['_dG'])/(1000*4.18)
                res_name = dict_aa[sequence]
                row_values ={
                            'id': f'{base_name[0]}_{replicate}_{res_name}{res_idx}',
                            'prot': base_name[0],
                            'replicate': replicate,
                            'res_name': res_name,
                            'res_idx': res_idx,
                            'dG': dG
                        }
                data_list.append(row_values)
    data = pd.DataFrame(data_list, columns=columns)
    return data


def rms_plots(dir,suffix,val):
    '''
    Function to create rms plots, if rmsd or rmsd is given by 'val ' parameter
    takes dictionary where all files with the descriptors are and takes suffix from the files name like '_descriptors_data.pkl'
    '''
    os.chdir(dir)
    files =sorted([f for f in os.listdir(dir) if f.endswith(suffix)])
    rms_data=[]
    names=[]
    averages=[]
    indexes=[]
    for file in files:
        with open(file, 'rb') as data:
            data_file=pickle.load(data)
        file_name_without_suffix = file.removesuffix(suffix)

        names.append(file_name_without_suffix)
        rms_data.append(data_file[val])
        averages.append(np.mean(data_file[val]))
        indexes.append(next((i for i, value in enumerate(data_file[val]) if value > 0.35), 0))

    n_plots = len(rms_data)  
    n_cols = 9  
    n_rows = math.ceil(n_plots / n_cols) 

   
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 25))
    axes = axes.flatten()  # Flatten the grid for easy indexing

    if val=='rmsd':
        
        vertical_line_handle = mlines.Line2D([], [], color='red',  linewidth=2,linestyle='--', label='Primer fotograma para rmsd  > 0.35nm')
        horizontal_line_handle = mlines.Line2D([], [], color='darkgreen',  linewidth=2.5,linestyle='-.', label='RMSD medio')

        for i, (rms, name, avg, index) in enumerate(zip(rms_data, names, averages, indexes)):
            if avg < 0.35:
                rmsd_color='darkblue'
            else:
                rmsd_color='blue'  
            axes[i].plot(rms, color=rmsd_color)  
            axes[i].set_title(name,fontsize=16) 
            axes[i].set_xlabel("Fotograma",fontsize=16)  
            axes[i].set_ylabel("nm",fontsize=16)  
            axes[i].grid(True) 
            
            axes[i].axvline(x=index, color='red', linestyle='--')  #
            
            axes[i].axhline(y=avg, color='darkgreen', linestyle='-.')  

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle('', fontsize=28, y=1.02)  
        fig.subplots_adjust(top=0.92)  
        plt.tight_layout(pad=3.5)  

        fig.legend(handles=[vertical_line_handle, horizontal_line_handle], loc='upper center', 
                fontsize=16, bbox_to_anchor=(0.5, 1), borderaxespad=0.5)
        plt.tight_layout(pad=3.0)  
        plt.savefig('rmsd_distribution.png', dpi=300,bbox_inches='tight')
        plt.show()
    else:

        for i, (rms, name, avg, index) in enumerate(zip(rms_data, names, averages, indexes)):
            axes[i].plot(rms)  
            axes[i].set_title(name,fontsize=16)  
            axes[i].set_xlabel("Fotograma",fontsize=16)  
            axes[i].set_ylabel("nm",fontsize=16)  
            axes[i].grid(True)  

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle('Evolución del RMSF para cada proteína', fontsize=28, y=1.02)  # Title for all plots
        fig.subplots_adjust(top=0.92)  
        plt.tight_layout(pad=3.5)  

        plt.tight_layout(pad=3.0)  
        plt.savefig('rmsf_distribution.png', dpi=300,bbox_inches='tight')
        plt.show()


def plots_regressions(X_bayesian,X_deuterium,Y_gibbs,Y_deuterium,Y_pred_gibbs_bayes,Y_pred_deuterium_bayes,Y_pred_gibbs_linear,Y_pred_deuterium_linear,score_type,time,replicate,base_name,results_r2):
    '''
    Plots regressions so it takes the X values for the bayesian and linear regression, the Y real values and the predictions
    'score type' is a string with the type of score
    'time' is interval of time used for doing the linear regressions with deuterium uptake
    'replicate' is the replicate
    'base_name' is the name of protein being processed
    'results_r2' list with r2 values

    '''
    # Plot the results
    # Create a plot
    sort_idx = np.argsort(X_bayesian[:, 0])  # Indices to sort based on x-axis
    X_sorted = X_bayesian[sort_idx, 0]
    Y_pred_sorted_gibbs = Y_pred_gibbs_bayes[sort_idx]
    Y_pred_sorted_deuterium = Y_pred_deuterium_bayes[sort_idx]
    # Use sorted values for plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    ax1, ax2,ax3,ax4= axes.flatten()  # Flatten the 2D array of axes to 1D and unpack
    # Plot Corrlation  bayesian and gibbs
    ax1.scatter(X_sorted, Y_gibbs, color='blue', label='Bayesian Prediction')
    ax1.plot(X_sorted,Y_pred_sorted_gibbs, color='red', label='Bayesian Prediction')
    ax1.set_xlabel('Puntuaciones puentes de hidrógeno ')
    ax1.set_ylabel(f'dG')
    ax1.set_title(f' Regresión bayesiana usando dG \n r2: {results_r2[0]}')
    ax1.legend()
    ax2.scatter(X_sorted, Y_deuterium, color='blue', label='Data')
    ax2.plot(X_sorted, Y_pred_sorted_deuterium, color='green', label='Bayesian Prediction')
    ax2.set_xlabel('Puntuaciones puentes de hidrógeno')
    ax2.set_ylabel(f'Deuterio incorporado')
    ax2.legend()
    ax2.set_title(f'Regresión bayesiana usando el Deuterio incorporado \n r2: {results_r2[1]}')
    # Adjust layout
    # Plot Corrlation  bayesian deuterium uptake
    ax3.scatter(X_deuterium, Y_gibbs, color='blue', label='Data')
    ax3.plot(X_deuterium,Y_pred_gibbs_linear, color='red', label='Linear Prediction')
    ax3.set_xlabel('Puntuación media de los puentes de hidrógeno')
    ax3.set_ylabel(f'dG')
    ax3.set_title(f' Regresión lineal usando dG y la media\n r2: {results_r2[2]}')

    ax3.legend()

    # Linear Regression
    ax4.scatter(X_deuterium, Y_deuterium, color='blue', label='Data')
    ax4.plot(X_deuterium, Y_pred_deuterium_linear, color='green', label='Linear Prediction')
    ax4.set_xlabel('Puntuación media de los puentes de hidrógeno')
    ax4.set_ylabel(f'Deuterio incorporado')
    ax4.legend()
    ax4.set_title(f' Regresión lineal usando dG y la media\n r2: {results_r2[3]}')
    ax4.set_ylim(0, 1)  # Set y-axis limits between 0 and 1
    # Adjust layout
    fig.tight_layout()
    plt.show()


def regression(X, Y, type_reg):
    # REGRESSIONS
    if type_reg == 'bayesian':
        # Fit Bayesian Ridge Regression
        bayesian_model = BayesianRidge()
        bayesian_model.fit(X, Y)
        Y_pred, Y_std = bayesian_model.predict(X, return_std=True)
    
    elif type_reg == 'linear':
        # Fit Linear Regression
        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(X, Y)
        Y_pred = linear_model.predict(X)
    
    else:
        raise ValueError("Unsupported regression type. Choose from 'linear', 'bayesian'")
    
    # Evaluate performance
    mse = mean_squared_error(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    results = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    
    if type_reg == 'bayesian':
        return results, Y_pred
    elif type_reg == 'linear':
        return results, Y_pred





def final_comparision_peptidev2(D_uptake,gibbs_processed,bonds,time_values,score_type,replicate,base_name,results_df,corr_data,range_val=10001):
    #Lists of mean hbonds scores values, all hbonds scores values('D_uptake_hbonds_bayesian'), Gibbs and HDX uptake
    gibbs_values_per_peptide=[]
    HDX_uptake_peptide=[]
    D_uptake_hbonds_bayesian=[]
    D_uptake_hbonds_mean=[]


    for time in time_values:
        for index, row in D_uptake.iterrows():
            HDX=row[time]
            start=row['start']
            end=row['end']
            values_bayesian=[]
            values_peptide=[]
            gibbs_peptide=[]
            peptide_mean=[]

            for i in range(start, end + 1):
                #Saving peptide info
                length=(end+1)-start
                if i in bonds and bonds[i].any() and i in gibbs_processed['quantity'].values: #info for bonds scores and gibbs is taken only for  residues wich have both values
                    mean_aa=float(np.mean(bonds[i]))
                    val = [bonds[i][k] for k in range(range_val)]
                    gibbs_0=gibbs_processed.loc[gibbs_processed["quantity"] == i]
                    gibbs_1 = gibbs_0['dG'].tolist()
                    gibbs_aa=gibbs_1[0]

                else:
                    mean_aa=0
                    val = [0] * range_val
                    gibbs_aa=0

                gibbs_peptide.append(gibbs_aa)
                values_bayesian.append(val)
                peptide_mean.append(mean_aa)

            #Peptide calculations

            # Scores of hbonds for all frames. For each peptide, for each frame the average of all residues is calculated
            #Each peptide is represented with values for all frames
            peptide_avg_bayesian= np.mean(values_bayesian, axis=0)
            D_uptake_hbonds_bayesian.append(peptide_avg_bayesian) #Saves each peptide info to a list

            #Mean hbonds scores. For each peptide, the average of the residue mean score is calculated to obtain a value per peptide
            avg_mean_peptide=np.mean(np.array(peptide_mean)) 
            D_uptake_hbonds_mean.append(avg_mean_peptide)#Saves each peptide info to a list

            # Gibbs values per petide. For each peptide, the average of the gibbs per residue is calculated to obtain a value per peptide
            gibbs_peptide = np.nan_to_num(gibbs_peptide, nan=0.0) #nan are converted to 0,cause no info
            avg_gibbs_peptide=np.mean(np.array(gibbs_peptide))
            gibbs_values_per_peptide.append(avg_gibbs_peptide)#Saves each peptide info to a list
            
            #Deuterium uptake per petide. This save the experimental value
            HDX_uptake_peptide.append(HDX)#Saves each peptide info to a list

        # Reshaping X and Y for sklearn 
        X_bayesian = np.array(D_uptake_hbonds_bayesian)
        X_mean=np.array(D_uptake_hbonds_mean).reshape(-1, 1)
        Y_gibbs_peptide=np.array(gibbs_values_per_peptide)
        Y_D_uptake=np.array(HDX_uptake_peptide)

        #Correlations calculations: Bayesian and Linear for both Gibbs values and peptide values

        bayesian_results_gibbs,bayesian_Y_pred_gibbs=regression(X_bayesian,Y_gibbs_peptide,'bayesian')
        bayesian_results_deu,bayesian_Y_pred_deu=regression(X_bayesian,Y_D_uptake,'bayesian')
        linear_results_gibbs_mean,linear_Y_pred_gibbs_mean=regression(X_mean,Y_gibbs_peptide,'linear')
        linear_results_deu_mean,linear_Y_pred_deu_mean=regression(X_mean,Y_D_uptake,'linear')

        #List with r2 values to use in plots
        results_r2=[bayesian_results_gibbs['r2'],bayesian_results_deu['r2'],linear_results_gibbs_mean['r2'],linear_results_deu_mean['r2']]
        #Plots comparing regressions
        #plots_regressions(X_bayesian,X_mean,Y_gibbs_peptide,Y_D_uptake,bayesian_Y_pred_gibbs,bayesian_Y_pred_deu,linear_Y_pred_gibbs_mean,linear_Y_pred_deu_mean,score_type,time,replicate,base_name,results_r2)
       
        #Saving results into dataframe
        
        files=[bayesian_results_gibbs,bayesian_results_deu,linear_results_gibbs_mean,linear_results_deu_mean]
        X_vals=['bayesian','bayesian','mean','mean']
        Y_vals = ['gibbs', 'D uptake'] * 2

        Y_value=[Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake]
        X_value=[X_bayesian,X_bayesian,X_mean,X_mean]
        for i in range(4):
            new_rows = pd.DataFrame([
            {'Protein': f'{base_name}', 'replicate': f'{replicate}', 'time_value': f'{time}', 'score_type':{score_type}, 'Y': Y_vals[i], 'Y_value':Y_value[i],'X': X_vals[i],'X_value':X_value[i], 'mse': files[i]['mse'], 'mae': files[i]['mae'], 'r2': files[i]['r2']}])

            # Concatenate with the existing DataFrame
            results_df= pd.concat([results_df, new_rows], ignore_index=True)

        data={}
        data['X_bayesian']=X_bayesian
        data['X_mean']=X_mean
        data['gibbs_peptide']=Y_gibbs_peptide
        data['deu_peptide']=Y_D_uptake
        data['gibbs_bayesian_pred']=bayesian_Y_pred_gibbs
        data['deu_bayesian_pred']=bayesian_Y_pred_deu
        data['gibbs_linear_pred']=linear_Y_pred_gibbs_mean
        data['deu_lienar_pred']=linear_Y_pred_deu_mean

        name=f'{base_name}_{score_type}_{replicate}'
        corr_data[name]=data


    return results_df,corr_data
        

def final_comparision_aa(D_uptake,gibbs_processed,bonds,time_values,score_type,replicate,base_name,results_df,corr_data):
    aa_seq=[]
    gibbs_values_res=[]
    mean_values=[]
    bayesian_values=[]
    median_values=[]
    for i in gibbs_processed['quantity'].values:
        if i in  bonds and bonds[i].any():
            gibbs_0=gibbs_processed.loc[gibbs_processed["quantity"] == i]
            gibbs_1 = gibbs_0['dG'].tolist()
            gibbs_aa=float(gibbs_1[0]/(1000*4.18))
            val = [bonds[i][k] for k in range(10001)]
            if np.isnan(gibbs_aa):
                continue
            else:        
                gibbs_values_res.append(gibbs_aa)
                mean_values.append(np.mean(bonds[i]))
                bayesian_values.append(val)
                median_values.append(np.median(bonds[i]))
                aa_seq.append(i)

    X_bayesian = np.array(bayesian_values)
    Y_gibbs_aa=np.array(gibbs_values_res)
    X_mean = np.array(mean_values).reshape(-1, 1)
    X_median =np.array(median_values).reshape(-1,1)
    bayesian_results_gibbs,bayesian_Y_pred_gibbs=regression(X_bayesian,Y_gibbs_aa,'bayesian')
    linear_results_gibbs_mean,linear_Y_pred_gibbs_mean=regression(X_mean,Y_gibbs_aa,'linear')
    linear_results_gibbs_median,linear_Y_pred_gibbs_median=regression(X_median,Y_gibbs_aa,'linear')

    r2_regs=[]
    for value in [bayesian_results_gibbs,linear_results_gibbs_mean,linear_results_gibbs_median]:
        r2_regs.append(value['r2'])
    files=[bayesian_results_gibbs,linear_results_gibbs_mean,linear_results_gibbs_median]
    X_vals=['bayesian','mean','median']
    Y_vals = ['gibbs'] * 3

    Y_value=[Y_gibbs_aa,Y_gibbs_aa,Y_gibbs_aa]
    X_value=[X_bayesian,X_mean,X_median]
    for i in range(3):
        new_rows = pd.DataFrame([
        {'Protein': f'{base_name}', 'replicate': f'{replicate}', 'score_type':{score_type}, 'Y': Y_vals[i], 'Y_value':Y_value[i],'X': X_vals[i],'X_value':X_value[i], 'mse': files[i]['mse'], 'mae': files[i]['mae'], 'r2': files[i]['r2']}])

        # Concatenate with the existing DataFrame
        results_df= pd.concat([results_df, new_rows], ignore_index=True)
    data={}
    data['X_bayesian']=X_bayesian
    data['X_mean']=X_mean
    data['X_median']=X_median
    data['gibbs_peptide']=Y_gibbs_aa
    data['gibbs_bayesian_pred']=bayesian_Y_pred_gibbs
    data['gibbs_mean_pred']=linear_Y_pred_gibbs_mean
    data['gibbs_median_pred']=linear_Y_pred_gibbs_median

    name=f'{base_name}_{score_type}_{replicate}'
    corr_data[name]=data
   
    
    return results_df, corr_data


def final_comparision_em(D_uptake,gibbs_processed,bonds,time_values,score_type,replicate,base_name,results_df,corr_data):
    # Extraction of Scores for the corresponding peptides
    D_uptake_hbonds_em=[]
    gibbs_values_per_peptide=[]
    HDX_uptake_peptide=[]
    for time in time_values:
        for index, row in D_uptake.iterrows():
            HDX=row[time]
            start=row['start']
            end=row['end']
            values_em=[]
            missing_peptide=[]
            missing=0
            gibbs_peptide=[]

            for i in range(start, end + 1):
                length=(end+1)-start
                if i in bonds and bonds[i].any() and i in gibbs_processed['quantity'].values:
                    val = float(bonds[i])
                    gibbs_0=gibbs_processed.loc[gibbs_processed["quantity"] == i]
                    gibbs_1 = gibbs_0['dG'].tolist()
                    gibbs_aa=gibbs_1[0]
                else:
                    val = 0                                                  
                    gibbs_aa=0
                gibbs_peptide.append(gibbs_aa)
                values_em.append(val)

            # Scores of hbonds em
            peptide_avg_em= np.mean(np.array(values_em))
            D_uptake_hbonds_em.append(peptide_avg_em)

            # Gibbs values per petide
            gibbs_peptide = np.nan_to_num(gibbs_peptide, nan=0.0)
            avg_gibbs_peptide=np.mean(np.array(gibbs_peptide))
            gibbs_values_per_peptide.append(avg_gibbs_peptide)
            
            #Deuterium uptake per petide
            HDX_uptake_peptide.append(HDX)
        # X and Y for sklearn
        X_em = np.array(D_uptake_hbonds_em).reshape(-1, 1)
        Y_gibbs_peptide=np.array(gibbs_values_per_peptide)
        Y_D_uptake=np.array(HDX_uptake_peptide)

        em_results_gibbs,em_Y_pred_gibbs=regression(X_em,Y_gibbs_peptide,'linear')
        em_results_deu,em_Y_pred_deu=regression(X_em,Y_D_uptake,'linear')


        files=[em_results_gibbs,em_results_deu]

        X_vals=['em','em']
        Y_vals = ['gibbs', 'D uptake']

        Y_value=[Y_gibbs_peptide,Y_D_uptake]
        X_value=[X_em,X_em]
        for i in range(2):
            new_rows = pd.DataFrame([
            {'Protein': f'{base_name}', 'replicate': f'{replicate}', 'time_value': f'{time}', 'score_type':{score_type}, 'Y': Y_vals[i], 'Y_value':Y_value[i],'X': X_vals[i],'X_value':X_value[i], 'mse': files[i]['mse'], 'mae': files[i]['mae'], 'r2': files[i]['r2']}])

            # Concatenate with the existing DataFrame
            results_df= pd.concat([results_df, new_rows], ignore_index=True)

        data={}
        data['X_em']=X_em
        data['gibbs_peptide']=Y_gibbs_peptide
        data['deu_peptide']=Y_D_uptake
        data['gibbs_em_pred']=em_Y_pred_gibbs
        data['deu_em_pred']=em_Y_pred_deu
   

        name=f'{base_name}_{score_type}_{replicate}'
        corr_data[name]=data

    return results_df,corr_data
        



def blocks(D_uptake,gibbs_processed,bonds,time_values,score_type,replicate,base_name,results_df,corr_data):
    # Extraction of Scores for the corresponding peptides
    D_uptake_hbonds_median=[]
    D_uptake_hbonds_bayesian=[]
    gibbs_values_per_peptide=[]
    HDX_uptake_peptide=[]

    D_uptake_hbonds_mean=[]
    D_uptake_hbonds_75=[]
    D_uptake_hbonds_iqr=[]
    D_uptake_hbonds_mode=[]
    D_uptake_hbonds_90=[]

    for time in time_values:
        for index, row in D_uptake.iterrows():
            HDX=row[time]
            start=row['start']
            end=row['end']
            values_bayesian=[]
            values_peptide=[]
            peptide_median=[]
            gibbs_peptide=[]


            for i in range(start, end + 1):
                length=(end+1)-start
                if i in bonds and bonds[i].any() and i in gibbs_processed['quantity'].values:
                    median_aa = float(np.median(bonds[i]))
                    mean_aa=float(np.mean(bonds[i]))
                    seventyfive=float(np.percentile(bonds[i],75))
                    iqr_val=float(iqr(bonds[i]))
                    val = [bonds[i][k] for k in range(10001)]
                    gibbs_0=gibbs_processed.loc[gibbs_processed["quantity"] == i]
                    gibbs_1 = gibbs_0['dG'].tolist()
                    gibbs_aa=gibbs_1[0]
                    mode_val= float(stats.mode(bonds[i])[0])
                    ninety=float(np.percentile(bonds[i],90))

                    

                else:
                    median_aa = 0
                    mean_aa=0
                    seventyfive=0
                    iqr_val=0
                    val = [0] * 10001
                    gibbs_aa=0
                    missing+=1
                    mode_val= 0
                    ninety=0
                gibbs_peptide.append(gibbs_aa)

                values_bayesian.append(val)
                peptide_median.append(median_aa)
                peptide_mean.append(mean_aa)
                peptide_75.append(seventyfive)
                peptide_iqr.append(iqr_val)
                peptide_mode.append(mode_val)
                peptide_90.append(ninety)
            # Scores of hbonds bayesian
            peptide_avg_bayesian= np.mean(values_bayesian, axis=0)
            D_uptake_hbonds_bayesian.append(peptide_avg_bayesian)

            #Scores of hbonds  mean of medians
            avg_median_peptide=np.mean(np.array(peptide_median)) 
            D_uptake_hbonds_median.append(avg_median_peptide)

            #extra scors

            avg_mean_peptide=np.mean(np.array(peptide_mean)) 
            D_uptake_hbonds_mean.append(avg_mean_peptide)
            avg_75_peptide=np.mean(np.array(peptide_75)) 
            D_uptake_hbonds_75.append(avg_75_peptide)
            avg_iqr_peptide=np.mean(np.array(peptide_iqr)) 
            D_uptake_hbonds_iqr.append(avg_iqr_peptide)

            avg_mode=np.mean(np.array(peptide_mode))
            D_uptake_hbonds_mode.append(avg_mode)
            avg_90=np.mean(np.array(peptide_90))
            D_uptake_hbonds_90.append(avg_90)
             

            # Gibbs values per petide
            gibbs_peptide = np.nan_to_num(gibbs_peptide, nan=0.0)
            avg_gibbs_peptide=np.mean(np.array(gibbs_peptide))
            gibbs_values_per_peptide.append(avg_gibbs_peptide)
            
            #Deuterium uptake per petide
            HDX_uptake_peptide.append(HDX)



        # X and Y for sklearn
        X_bayesian = np.array(D_uptake_hbonds_bayesian)
        median_values = np.array(D_uptake_hbonds_median).reshape(-1, 1)
        mean_values=np.array(D_uptake_hbonds_mean).reshape(-1, 1)
        seventyfive_values=np.array(D_uptake_hbonds_75).reshape(-1, 1)
        iqr_values=np.array(D_uptake_hbonds_iqr).reshape(-1, 1)
        mode_values=np.array(D_uptake_hbonds_iqr).reshape(-1, 1)
        ninety_values=np.array(D_uptake_hbonds_90).reshape(-1, 1)

        Y_gibbs_peptide=np.array(gibbs_values_per_peptide)
        Y_D_uptake=np.array(HDX_uptake_peptide)


        files=[em_results_gibbs,em_results_deu]

        X_vals=['em','em']
        Y_vals = ['gibbs', 'D uptake'] 
        for i in range(2):
            new_rows = pd.DataFrame([
            {'Protein': f'{base_name}', 'replicate': f'{replicate}', 'time_value': f'{time}', 'score_type':{score_type}, 'Y': Y_vals[i], 'X': X_vals[i], 'mse': files[i]['mse'], 'mae': files[i]['mae'], 'r2': files[i]['r2']}])

            # Concatenate with the existing DataFrame
            results_df= pd.concat([results_df, new_rows], ignore_index=True)

        
  
        data  = pd.DataFrame({
             'Uptake': Y_D_uptake,
             'Gibbs': Y_gibbs_peptide,
              'em': X_em.flatten(),

            })

        name=f'{base_name}_{score_type}_{replicate}'
        corr_data[name]=data
    return results_df,corr_data



def comparasion_parameter(D_uptake,gibbs_processed,bonds,time_values,score_type,replicate,base_name,results_df,corr_data):
    
    # Extraction of Scores for the corresponding peptides
    gibbs_values_per_peptide=[]
    HDX_uptake_peptide=[]
    D_uptake_hbonds_median=[]
    D_uptake_hbonds_mean=[]
    D_uptake_hbonds_75=[]
    D_uptake_hbonds_iqr=[]
    D_uptake_hbonds_mode=[]
    D_uptake_hbonds_90=[]

    for time in time_values:
        for index, row in D_uptake.iterrows():
            HDX=row[time]
            start=row['start']
            end=row['end']
            values_peptide=[]
            peptide_median=[]
            gibbs_peptide=[]
            peptide_mean=[]
            peptide_75=[]
            peptide_iqr=[]
            missing_peptide=[]
            missing=0
            peptide_mode=[]
            peptide_90=[]


            for i in range(start, end + 1):
                length=(end+1)-start
                if i in bonds and bonds[i].any() and i in gibbs_processed['quantity'].values:
                    median_aa = float(np.median(bonds[i]))
                    mean_aa=float(np.mean(bonds[i]))
                    seventyfive=float(np.percentile(bonds[i],75))
                    iqr_val=float(iqr(bonds[i]))
                    gibbs_0=gibbs_processed.loc[gibbs_processed["quantity"] == i]
                    gibbs_1 = gibbs_0['dG'].tolist()
                    gibbs_aa=gibbs_1[0]
                    mode_val= float(stats.mode(bonds[i])[0])
                    ninety=float(np.percentile(bonds[i],90))


                else:
                    median_aa = 0
                    mean_aa=0
                    seventyfive=0
                    iqr_val=0
                    gibbs_aa=0
                    missing+=1
                    mode_val= 0
                    ninety=0
                gibbs_peptide.append(gibbs_aa)
                peptide_median.append(median_aa)
                peptide_mean.append(mean_aa)
                peptide_75.append(seventyfive)
                peptide_iqr.append(iqr_val)
                peptide_mode.append(mode_val)
                peptide_90.append(ninety)
      
            #Scores of hbonds 
            avg_median_peptide=np.mean(np.array(peptide_median)) 
            D_uptake_hbonds_median.append(avg_median_peptide)
            avg_mean_peptide=np.mean(np.array(peptide_mean)) 
            D_uptake_hbonds_mean.append(avg_mean_peptide)
            avg_75_peptide=np.mean(np.array(peptide_75)) 
            D_uptake_hbonds_75.append(avg_75_peptide)
            avg_iqr_peptide=np.mean(np.array(peptide_iqr)) 
            D_uptake_hbonds_iqr.append(avg_iqr_peptide)

            avg_mode=np.mean(np.array(peptide_mode))
            D_uptake_hbonds_mode.append(avg_mode)
            avg_90=np.mean(np.array(peptide_90))
            D_uptake_hbonds_90.append(avg_90)


            # Gibbs values per petide
            gibbs_peptide = np.nan_to_num(gibbs_peptide, nan=0.0)
            avg_gibbs_peptide=np.mean(np.array(gibbs_peptide))
            gibbs_values_per_peptide.append(avg_gibbs_peptide)
            
            #Deuterium uptake per petide
            HDX_uptake_peptide.append(HDX)



        # X and Y for sklearn
        median_values = np.array(D_uptake_hbonds_median).reshape(-1, 1)
        mean_values=np.array(D_uptake_hbonds_mean).reshape(-1, 1)
        seventyfive_values=np.array(D_uptake_hbonds_75).reshape(-1, 1)
        iqr_values=np.array(D_uptake_hbonds_iqr).reshape(-1, 1)
        mode_values=np.array(D_uptake_hbonds_iqr).reshape(-1, 1)
        ninety_values=np.array(D_uptake_hbonds_90).reshape(-1, 1)

        Y_gibbs_peptide=np.array(gibbs_values_per_peptide)
        Y_D_uptake=np.array(HDX_uptake_peptide)

  
        #Results and predicitons for the correlations
        #median
        linear_results_gibbs_median,linear_Y_pred_gibbs_median=regression(median_values,Y_gibbs_peptide,'linear')
        linear_results_deu_median,linear_Y_pred_deu_median=regression(median_values,Y_D_uptake,'linear')
        #mean
        linear_results_gibbs_mean,linear_Y_pred_gibbs_mean=regression(mean_values,Y_gibbs_peptide,'linear')
        linear_results_deu_mean,linear_Y_pred_deu_mean=regression(mean_values,Y_D_uptake,'linear')
        #quartile 75
        linear_results_gibbs_75,linear_Y_pred_gibbs_75=regression(seventyfive_values,Y_gibbs_peptide,'linear')
        linear_results_deu_75,linear_Y_pred_deu_75=regression(seventyfive_values,Y_D_uptake,'linear')
        #q¡interquartile
        linear_results_gibbs_iqr,linear_Y_pred_gibbs_iqr=regression(iqr_values,Y_gibbs_peptide,'linear')
        linear_results_deu_iqr,linear_Y_pred_deu_iqr=regression(iqr_values,Y_D_uptake,'linear')
        #mode
        linear_results_gibbs_mode,linear_Y_pred_gibbs_mode=regression(mode_values,Y_gibbs_peptide,'linear')
        linear_results_deu_mode,linear_Y_pred_deu_mode=regression(mode_values,Y_D_uptake,'linear')
        #quartile 90
        linear_results_gibbs_90,linear_Y_pred_gibbs_90=regression(ninety_values,Y_gibbs_peptide,'linear')
        linear_results_deu_90,linear_Y_pred_deu_90=regression(ninety_values,Y_D_uptake,'linear')


        files=[linear_results_gibbs_median,linear_results_deu_median,linear_results_gibbs_mean,linear_results_deu_mean,linear_results_gibbs_75,linear_results_deu_75,linear_results_gibbs_90,linear_results_deu_90,linear_results_gibbs_iqr,linear_results_deu_iqr,linear_results_gibbs_mode,linear_results_deu_mode]

        X_vals=['median','median','mean','mean','quartile 75','quartile 75','quartile 90','quartile 90','interquartile','interquartile','mode','mode']
        Y_vals = ['gibbs', 'D uptake'] * 6
        Y_value=[Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake,Y_gibbs_peptide,Y_D_uptake]
        X_value=[median_values,median_values,mean_values,mean_values,seventyfive_values,seventyfive_values,ninety_values,ninety_values,iqr_values,iqr_values,mode_values,mode_values]
        for i in range(12):
            new_rows = pd.DataFrame([
            {'Protein': f'{base_name}', 'replicate': f'{replicate}', 'time_value': f'{time}', 'score_type':{score_type}, 'Y': Y_vals[i], 'Y_value':Y_value[i],'X': X_vals[i],'X_value':X_value[i], 'mse': files[i]['mse'], 'mae': files[i]['mae'], 'r2': files[i]['r2']}])

            # Concatenate with the existing DataFrame
            results_df= pd.concat([results_df, new_rows], ignore_index=True)
        
        
  
        data  = pd.DataFrame({
             'Uptake': Y_D_uptake,
             'Gibbs': Y_gibbs_peptide,
            'median': median_values.flatten(),
             'mean':mean_values.flatten(),
             '75':seventyfive_values.flatten(),
             '90':ninety_values.flatten(),
             'iqr':iqr_values.flatten(),
             'mode':mode_values.flatten()
            })

        name=f'{base_name}_{score_type}_{replicate}'
        corr_data[name]=data
        
        

    return results_df,corr_data
