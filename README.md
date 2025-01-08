# HDX-MS and MD integration
This is the GitHub repository associated with my master's thesis "CStudy of protein structural dynamics through the integration of bioinformatics methods and hydrogen-deuterium exchange mass spectrometry (HDX-MS) ". The purpose of this repository is to provide the code needed to replicate the results of this master's thesis. 

### Requisities
GROMACS,acpype and obabel were installed locally. For the rest of the packages conda environments were used. Due to uncompatibilities 3 different environments were created:
* HDX_MD to generate the MD and the associated data
* tsfresh to extract features and generate the models
* ml to analyse the models and use xgboost
To clone this environments .yml files are provided

### Data
Data is provided in this repository as well as in the ZENODO repository due to space limits within github
* params includes the parameters files needed within GROMACS to generate the dynamics
* HDX_deuterium_data includes was installed locally, for the rest of programs
* gibbs_data includes the files with the Gibbs energy
* Models includes the models and the training input except for the dataframe with the original dataframe and the training features.
  This due to their size have been including in the ZENODO repository where the bonds and descriptors for each file have been stored
  (https://zenodo.org/records/14618543?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxNTYxMTdhLTNlNzItNDc2Ni1iODQxLWM3ZWVjY2E4N2Y5OSIsImRhdGEiOnt9LCJyYW5kb20iOiIyZDFiY2Y0ZmViMmYyNTQ3NzE2M2EwN2Q1OTZhMjcyZiJ9.Y6AuC2Fw3ppJt06glIdnUYDg7KWeI64w1XhxlgXFqtgqyU3EUUDsOUcU_HT1Ih-1k9WryoPgqXf7NqHZUG-trw)

### Scripts
Scripts are store within 3 folders. If _q in the name, then the scripts are meant to be sent to queue
* Molecular dynamics. This scripts produce the simulations
* Descriptors. This generate the bonds and the rest of the descriptors
* Analysis. This stores the creation of the dataframe, the transformation of the HDX data to gibbs, and the .py files with all the funcions

### Results
In the main 3 jupyter notebooks contain
* the analysis done for the descriptors (hbonds punctuaction, rmsd... ---- 'preliminar_analysis'
* the models creation ---- models_final
* the models analysis ---- analisis_modelos
  
