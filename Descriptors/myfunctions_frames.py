import math
from scipy.spatial.distance import cdist
import numpy as np
from scipy.integrate import quad


def reassign_water_indices(df):
    '''Reassign_water_indices modifies indices of HOH since they are not properly indexed
    This way the residue index of the HOH atoms is the same'''

    # Creates a mask for water molecules
    water_mask = df['residue'] == 'HOH'
    water_df = df[water_mask].copy()

    # Initializes a new residue index for water molecules
    new_residue_index = df['residue_index'].max() + 1

    # Iterates through the water molecules and reassign indices
    for i in range(0, len(water_df), 3):
        water_df.iloc[i:i+3, water_df.columns.get_loc('residue_index')] = new_residue_index
        new_residue_index += 1

    # Updates the original DataFrame with the new indices
    df.update(water_df)
    return df

def update_ion_indices(df):
    '''update_ion_indices modifies indices of CL and NA since after updating HOH they stayed withthe old ones'''
    ion_mask = df['residue'].isin(['CL', 'NA'])  # Adjust this list based on your ion types
    ion_df = df[ion_mask].copy()

    # Initialize a new residue index for ions
    new_residue_index = df['residue_index'].max() + 1

    # Reassign indices for ions, incrementing by 1 from the previous row's index
    for i in range(len(ion_df)):
        ion_df.iloc[i, ion_df.columns.get_loc('residue_index')] = new_residue_index
        new_residue_index += 1
    
    # Update the original DataFrame with the new indices for ions
    df.update(ion_df)
    return df



def extract_coordenates_water(df,coordsN,coordsH,coordsAc,coordsX,coordsWAT_O,coordsWAT_H):
    """Extracts coordenates for the possible N-H atoms taking part in the bond as well as all possible acceptors"""
        coordenates_N_frame = []
        coordenates_H_frame = []
        coordenates_Ac_frame = []
        coordenates_X_frame = []
        coordenates_wat_O_frame =[]    
        coordenates_wat_H1_frame =[] 
        coordenates_wat_H2_frame = []
    
        for k in coordsN:
            coordenate_N = df.loc[k,['x', 'y', 'z']].values.tolist()
            coordenates_N_frame.append(coordenate_N)
        for j in coordsH:
            coordenate_H = df.loc[j,['x', 'y', 'z']].values.tolist()
            coordenates_H_frame.append(coordenate_H)
        for h in coordsAc:
            coordenate_Ac = df.loc[h,['x', 'y', 'z']].values.tolist()
            coordenates_Ac_frame.append(coordenate_Ac)
        for f in coordsX:
            coordenate_X = df.loc[f,['x', 'y', 'z']].values.tolist()
            coordenates_X_frame.append(coordenate_X)
        for v in coordsWAT_O:
            coordenate_wat_O = df.loc[v,['x', 'y', 'z']].values.tolist()
            coordenates_wat_O_frame.append(coordenate_wat_O)
        for p in coordsWAT_H:
            val1,val2 = p
            coordenate_wat_H1 = df.loc[val1,['x', 'y', 'z']].values.tolist()
            coordenate_wat_H2 = df.loc[val2,['x', 'y', 'z']].values.tolist()
            coordenates_wat_H1_frame.append(coordenate_wat_H1)
            coordenates_wat_H2_frame.append(coordenate_wat_H2)

        return coordenates_N_frame,coordenates_H_frame,coordenates_Ac_frame,coordenates_X_frame,coordenates_wat_O_frame,coordenates_wat_H1_frame,coordenates_wat_H2_frame

def distance_element(coords1,coords2):
    '''Distance_element calculates the distance between 2 arrays with coordenates. When they do have the same number of elements
    plus the calculation is done element by element'''

    # Checks same length
    assert len(coords1) == len(coords2)

    # Function to calculate the Euclidean distance between two points
    def euclidean_distance(point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    # Calculates distances between corresponding pairs of coordinates
    distances = [euclidean_distance(p1, p2) for p1, p2 in zip(coords1, coords2)]
    return distances

def vector_element(coords1,coords2):
    '''vector_element calculates the vector between 2 arrays with coordenates. When they do have the same number of elements
    plus the calculation is done element by element'''
    assert len(coords1) == len(coords2)
    
    def vector_making(point1, point2):
        return [a - b for a, b in zip(point1, point2)]
    
    vector = [vector_making(p1,p2) for p1,p2 in zip(coords1,coords2)] 
    return vector

def distance_mx(coords1,coords2):
    '''distance_mx  calculates the eucledian distance between 2 arrays with coordenates. It calculates the distance between all elements
    output is an array'''
    distances = cdist(coords1, coords2, metric='euclidean')
    return distances

def vector_mx(coords1,coords2):
    '''vector_mx  calculates the vectors between 2 arrays with coordenates. It calculates the vectos between all elements
    output is an array'''
    def vector_making(point1, point2):
        return [a - b for a, b in zip(point1, point2)]
    matrix  = np.zeros(shape=(len(coords1),len(coords2),3))

    for i in range(len(coords1)):
        for k in range(len(coords2)):
            vector = vector_making(coords1[i],coords2[k])
            matrix[i,k]=vector
    return matrix
  

def angle_alpha(nh,H_Ac):
    '''Input vectores N-H and H-Ac.
    Calculates de  angle between an array of vectors of size n and an array of vectors of size n x m'''
    matrix  = np.zeros(shape=(H_Ac.shape[0],H_Ac.shape[1]))
    for i in range(H_Ac.shape[0]):
        for k in range(H_Ac.shape[1]):
            dotproduct = np.dot(nh[i], H_Ac[i,k])
            magnitude_v1=math.sqrt(sum(x**2 for x in nh[i]))
            magnitude_v2=math.sqrt(sum(x**2 for x in H_Ac[i,k]))
            cos = dotproduct / (magnitude_v1*magnitude_v2)
            angles_rad = math.acos(cos)
            angles_deg = (math.degrees(angles_rad))%360
            matrix[i,k]=angles_deg
    return matrix

def angle_beta(AcX,H_Ac):
    '''Calculates de  angle between an array of vectors of size m and an array of vectors of size n x m'''
    matrix  = np.zeros(shape=(H_Ac.shape[0],H_Ac.shape[1]))
    for i in range(H_Ac.shape[0]):
        for k in range(H_Ac.shape[1]):
            dotproduct = np.dot(AcX[k], H_Ac[i,k])
            magnitude_v1=math.sqrt(sum(x**2 for x in AcX[k]))
            magnitude_v2=math.sqrt(sum(x**2 for x in H_Ac[i,k]))
            cos = dotproduct / (magnitude_v1*magnitude_v2)
            angles_rad = math.acos(cos)
            angles_deg = (math.degrees(angles_rad))%360
            matrix[i,k]=angles_deg
    return matrix

def indices_hbond(distance_dict,threshold):
    '''This function extracts the indices of the matrix distance where the value is lower than the threshold. It returns a dictionary with the indices per frame'''
    indices = np.where(distance_dict <  threshold)
    row_val,col_val=indices 
    pairs = list(zip(row_val, col_val))
    return pairs

def get_dis_angles(distance_HAc,alpha_angle,row,col,beta_angle1,beta_angle2=None):
    distance = distance_HAc[row,col]
    alpha = alpha_angle[row,col]
    beta_1 = beta_angle1[row,col]
    if beta_angle2 is not None and isinstance(beta_angle2, np.ndarray):
        beta_2 = beta_angle2[row,col]
        return distance,alpha,beta_1,beta_2
    else:
        return distance,alpha,beta_1

def chemscore(distance_hbond,alpha_hbond,beta_hbond, gaussian=False,r_ideal = 0.185, delta_r_ideal = 0.025, delta_r_max = 0.065, hbond_r_sigma = 0.1,
            alpha_ideal = 180, delta_alpha_ideal = 30, delta_alpha_max = 80 , hbond_alpha_sigma = 10,
            beta_ideal = 180, delta_beta_ideal = 70, delta_beta_max = 80, hbond_beta_sigma = 10):
    '''Calculates the Chemscore function'''
    # Input, distance anf alpha and beta angles
    delta_r = np.abs(distance_hbond - r_ideal)  # delta_r is The absolute deviation of the actual H..A separation from r_ideal
    delta_alpha = np.abs(alpha_hbond - alpha_ideal) # delta_alpha is the absolute deviation of the actual D-H..A angle from alpha_ideal
    delta_beta = np.abs(beta_hbond - beta_ideal) # delta_beta The absolute deviation of the actual H..A-X angle from β
    
    def gauss(u, sigma): 
        return np.exp(-(u) ** 2 / (2 * sigma ** 2))

    def B(x,x_ideal,x_max):
        if x <= x_ideal:
            return 1
        elif x_ideal < x <= x_max:
            return 1-((x-x_ideal)/(x_max-x_ideal))  
        elif x > x_max:
            return 0
        
    def integrand_B_value(u, x, x_ideal, x_max, sigma):
        return B(x - u, x_ideal, x_max) * gauss(u, sigma)
    
    def integrand_gauss_value(u, sigma):
        return gauss(u, sigma) 

    def B_prime(x,x_ideal,x_max,sigma):
        num, _= quad(integrand_B_value, -np.inf, np.inf, args=(x, x_ideal, x_max, sigma))
        den, _ = quad(integrand_gauss_value, -np.inf, np.inf, args=(sigma,))
        return num / den
    
    if gaussian:
        par1 = B_prime(delta_r,delta_r_ideal,delta_r_max,hbond_r_sigma)
        par2 = B_prime(delta_alpha,delta_alpha_ideal,delta_alpha_max,hbond_alpha_sigma) 
        par3 = B_prime(delta_beta,delta_beta_ideal,delta_beta_max,hbond_beta_sigma) 
    else:
        par1 = B(delta_r,delta_r_ideal,delta_r_max)
        par2 = B(delta_alpha,delta_alpha_ideal,delta_alpha_max) 
        par3 = B(delta_beta,delta_beta_ideal,delta_beta_max) 

    G_score = par1 * par2 * par3
    return G_score

def chemscore_water(distance_hbond,alpha_hbond,beta1_hbond,beta2_hbond,gaussian=False, r_ideal = 0.185, delta_r_ideal = 0.025, delta_r_max = 0.065, hbond_r_sigma = 0.1,
            alpha_ideal = 180, delta_alpha_ideal = 30, delta_alpha_max = 80 , hbond_alpha_sigma = 10,
            beta_ideal = 140, delta_beta_ideal = 30, delta_beta_max = 40, hbond_beta_sigma = 10):
    '''Calculates the Chemscore function for water. It takes into account  both hydrogens attached to the oxygen of water'''
    # Input, distance anf alpha and beta angles
    delta_r = np.abs(distance_hbond - r_ideal)  # delta_r is The absolute deviation of the actual H..A separation from r_ideal
    delta_alpha = np.abs(alpha_hbond - alpha_ideal) # delta_alpha is the absolute deviation of the actual D-H..A angle from alpha_ideal
    delta_beta1 = np.abs(beta1_hbond - beta_ideal) # delta_beta The absolute deviation of the actual H..A-X angle from β
    delta_beta2 = np.abs(beta2_hbond - beta_ideal) # delta_beta The absolute deviation of the actual H..A-X angle from β

    def gauss(u, sigma): 
        return np.exp(-(u) ** 2 / (2 * sigma ** 2))
    
    def B(x,x_ideal,x_max):
        if x <= x_ideal:
            return 1
        elif x_ideal < x <= x_max:
            return 1-((x-x_ideal)/(x_max-x_ideal))  
        elif x > x_max:
            return 0
        
    def integrand_B_value(u, x, x_ideal, x_max, sigma):
        return B(x - u, x_ideal, x_max) * gauss(u, sigma)
    
    def integrand_gauss_value(u, sigma):
        return gauss(u, sigma) 

    def B_prime(x,x_ideal,x_max,sigma):
        num, _= quad(integrand_B_value, -np.inf, np.inf, args=(x, x_ideal, x_max, sigma))
        den, _ = quad(integrand_gauss_value, -np.inf, np.inf, args=(sigma,))
        return num / den
    if gaussian:
        par1 = B_prime(delta_r,delta_r_ideal,delta_r_max,hbond_r_sigma)
        par2 = B_prime(delta_alpha,delta_alpha_ideal,delta_alpha_max,hbond_alpha_sigma) 
        par3 = B_prime(delta_beta1,delta_beta_ideal,delta_beta_max,hbond_beta_sigma) * B_prime(delta_beta2,delta_beta_ideal,delta_beta_max,hbond_beta_sigma)
    else:
        par1 = B(delta_r,delta_r_ideal,delta_r_max)
        par2 = B(delta_alpha,delta_alpha_ideal,delta_alpha_max) 
        par3 = B(delta_beta1,delta_beta_ideal,delta_beta_max) * B(delta_beta2,delta_beta_ideal,delta_beta_max) 

    G_score = par1 * par2 * par3
    return G_score

def get_atoms(pair,indices_H,indices_Ac):
    '''Gets atom number and residue name for the atoms part of a hydrogen bonds'''
    row,col = pair
    H_atom = indices_H[row]
    Ac_atom = indices_Ac[col]
    return H_atom,Ac_atom


def electrostatic(N_charge_idx,H_hbond_idx,Ac_hbond_idx,H_charge_idx,Ac_charge_idx,top_df,distance_HAc,distance_NAc_vals):
    """This function calculates the electrostatic score.
    It takes as input:
    Idexes of N,H and Ac to extract the charge from the dataframe top_df which includes the charge of each atom
    Indexes of H and Ac to extract the distance it is different than the other one because th edistance matrixes do not contained all atoms.
    Distances matrixes """
    charge_N = top_df.loc[N_charge_idx,'charge']
    charge_Ac = top_df.loc[Ac_charge_idx,'charge']
    charge_H = top_df.loc[H_charge_idx,'charge']
    dis_HAc_value = distance_HAc[H_hbond_idx,Ac_hbond_idx]
    dist_NAc_value = distance_NAc_vals[H_hbond_idx,Ac_hbond_idx]
    E = ((charge_H * charge_Ac)/dis_HAc_value) +((charge_N* charge_Ac)/dist_NAc_value)
    return E
