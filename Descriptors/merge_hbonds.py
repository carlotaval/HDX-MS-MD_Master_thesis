import os
import glob
import pickle
import argparse
import re
from collections import defaultdict, OrderedDict
'''This script merges the bonds calculated per frame into just one dictionary'''
def main():
        # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze molecular dynamics hbonds.')
    parser.add_argument('--directory', type=str, required=True, help='Directory where hbonds dictionaries are')
    parser.add_argument('--base_name', type=str, required=True, help='Base name for output files or directories')
    parser.add_argument('--outputdir', type=str, required=True, help='outputdir')

    args = parser.parse_args()
    
    os.chdir(args.directory)
    files=glob.glob(f'{args.base_name}_bonds_*')
    files.sort(key=lambda x: int(re.search(r'(\d+)', x.split('_')[-1].split('.')[0]).group()))

    electrostatic=defaultdict(list)
    geometric=defaultdict(list)
    geometric_gauss=defaultdict(list)

    #Load data from frames dicitonaries 
    for file_name in files:
        frame=int(re.search(r'(\d+)', file_name.split('_')[-1].split('.')[0]).group())
        with open(file_name,'rb') as file:
            data=pickle.load(file)
        electrostatic[frame].append(data['electrostatic'])
        geometric[frame].append(data['geometric'])
        geometric_gauss[frame].append(data['geometric_gauss'])
    for key in electrostatic:
        electrostatic[key] = [item for sublist in electrostatic[key] for item in sublist]
    for key in geometric:
        geometric[key] = [item for sublist in geometric[key] for item in sublist]
    for key in geometric_gauss:
        geometric_gauss[key] = [item for sublist in geometric_gauss[key] for item in sublist]
    
    
    #Save data in dictionary, where keys are donors and values are lists of [frame, aceptor,score
    electrostatic_final = defaultdict(list)
    electrostatic_final_water = defaultdict(list)
    for frame, hbonds in electrostatic.items():
        for donor, acceptor, score,type in hbonds:
            if type=='prot':
                electrostatic_final[donor].append((frame,acceptor, score))
            elif type=='wat':
                electrostatic_final_water[donor].append((frame,acceptor, score))
    electrostatic_final = OrderedDict(sorted(electrostatic_final.items()))
    electrostatic_final_water = OrderedDict(sorted(electrostatic_final_water.items()))

    geometric_final = defaultdict(list)
    geometric_final_water = defaultdict(list)

    for frame, hbonds in geometric.items():
        for donor, acceptor, score, type in hbonds:
            if score > 0.1:
                if type=='prot':
                    geometric_final[donor].append((frame,acceptor,score))
                elif type=='wat':
                    geometric_final_water[donor].append((frame,acceptor,score))
    geometric_final = OrderedDict(sorted(geometric_final.items()))
    geometric_final_water = OrderedDict(sorted(geometric_final_water.items()))

    geometric_gauss_final = defaultdict(list)
    geometric_gauss_final_water = defaultdict(list)

    for frame, hbonds in geometric_gauss.items():
        for donor, acceptor, score,type in hbonds:
            if score > 0.1:
                if type=='prot':
                    geometric_gauss_final[donor].append((frame,acceptor, score))
                elif type=='wat':
                    geometric_gauss_final_water[donor].append((frame,acceptor, score))

    geometric_gauss_final = OrderedDict(sorted(geometric_gauss_final.items()))
    geometric_gauss_final_water = OrderedDict(sorted(geometric_gauss_final_water.items()))

    #Save into final dicitonary
    bonds_prot={}
    bonds_prot['electrostatic'] = electrostatic_final
    bonds_prot['geometric'] = geometric_final
    bonds_prot['geometric_gauss'] = geometric_gauss_final

    bonds_water={}
    bonds_water['electrostatic'] = electrostatic_final_water
    bonds_water['geometric'] = geometric_final_water
    bonds_water['geometric_gauss'] = geometric_gauss_final_water
    os.chdir(args.outputdir)

    output_file = f'{args.base_name}_bonds_prot.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(bonds_prot, f)

        output_file = f'{args.base_name}_bonds_water.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(bonds_water, f)

if __name__ == '__main__':
    main()
