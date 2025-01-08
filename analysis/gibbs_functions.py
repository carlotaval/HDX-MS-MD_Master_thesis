from pyhdx import read_dynamx, HDXMeasurement
from pyhdx.fitting import (
    fit_rates_half_time_interpolate,
    fit_rates_weighted_average,
    fit_gibbs_global,
)
from pyhdx.process import filter_peptides, apply_control, correct_d_uptake
from pathlib import Path
import os

'''
This functions compute the gibbs calculations from the experimental data. compute_dG_NC when no control used in experimental data and  compute_dG_control when control was included
This file was implemented within HDX to dG notebook
'''

 
def compute_dG_NC(file_input,outputdir,sequence_prot,state_prot,output_name):
    
    data = read_dynamx(file_input)
     
    peptides = filter_peptides(data, state=state_prot)  # , query=["exposure != 0."])
    
    peptides['rfu'] = peptides['uptake']
     
    peptides_corrected = correct_d_uptake(peptides)
     
    sequence = sequence_prot
     
     
    hdxm = HDXMeasurement(peptides_corrected, temperature=303.15, pH=8.0, sequence=sequence)
     
     
    fr_half_time = fit_rates_half_time_interpolate(hdxm)
    print(fr_half_time.output)
     
    gibbs_guess = hdxm.guess_deltaG(fr_half_time.output["rate"])
     
    gibbs_result = fit_gibbs_global(hdxm, gibbs_guess, epochs=1000)
    gibbs_output = gibbs_result.output
    name=output_name+'_gibbs.csv'
    os.chdir(outputdir)
    gibbs_output.round(2).to_csv(name)

def compute_dG_control(file_input,outputdir,sequence_prot,state_prot,output_name,value):
  
    data = read_dynamx(file_input)
                       
    fd = {"state": "Full deuteration control", "exposure": {"value": value, "unit": "min"}}
    fd_df = filter_peptides(data, **fd)
    
    peptides = filter_peptides(data, state=state_prot)  # , query=["exposure != 0."])
    peptides_control = apply_control(peptides, fd_df)
    peptides_corrected = correct_d_uptake(peptides_control)
     
     
    sequence = sequence_prot
     
     
    hdxm = HDXMeasurement(peptides_corrected, temperature=303.15, pH=8.0, sequence=sequence)
     
     
    fr_half_time = fit_rates_half_time_interpolate(hdxm)
    print(fr_half_time.output)
     
    gibbs_guess = hdxm.guess_deltaG(fr_half_time.output["rate"])
     
    gibbs_result = fit_gibbs_global(hdxm, gibbs_guess, epochs=1000)
    gibbs_output = gibbs_result.output
    name=output_name+'_gibbs.csv'
    os.chdir(outputdir)
    gibbs_output.round(2).to_csv(name)
