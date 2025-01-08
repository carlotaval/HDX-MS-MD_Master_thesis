from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.dummy import DummyRegressor
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.feature_extraction import MinimalFCParameters, extract_features

def evaluate_model(X, y, model_name='knn', n_repeats=10, var_sel='No'):
    '''This function evaluates a machine learning model using cross-validation and hyperparameter tuning. It performs repeated 
	k-fold cross-validation to estimate the prediction error, and selects the best model based on the minimum error. 
	It supports different types of models (K-Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosting), 
	and allows for variable selection (optional), providing the average and standard deviation of prediction errors 
	as well as the best trained model.'''
    pred_error_list = []
    outer_cv = RepeatedKFold(n_splits=10, n_repeats=n_repeats, random_state=0)
    
    if model_name == 'knn':
        model = KNeighborsRegressor()
        param_grid = {'model__n_neighbors': [1, 3, 5, 7, 11, 15, 20, 25, 30, 35]}
    elif model_name == 'DecisionTree':
        model = DecisionTreeRegressor(random_state=0)
        param_grid = {'model__criterion':['squared_error']}
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=0)
        param_grid = {'model__criterion':['squared_error']}

    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=0)
        param_grid = {'model__criterion':['friedman_mse']}
        
    results = Parallel(n_jobs=8)(delayed(evaluate_fold)(
        train_idx, test_idx, X, y, var_sel, model_name, model, param_grid
    ) for train_idx, test_idx in outer_cv.split(X, y))
    
    if results:
        pred_error_list, models = zip(*results)
    
    average_pred_error = np.mean(pred_error_list)
    sd_pred_error = np.std(pred_error_list)
    # Train the final model on the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if len(pred_error_list) > 0:
        best_model_index = np.argmin(pred_error_list)
        best_model = models[best_model_index]
        best_model.fit(X_scaled, y)

    return average_pred_error, sd_pred_error, best_model


def evaluate_fold(train_idx, test_idx, X, y, var_sel, model_name, model, param_grid):
    '''
    This function performs evaluation of a machine learning model for a single fold of cross-validation. 
    It handles the scaling of input data, applies different feature selection methods (none, ANOVA, or Random Forest),
    and performs hyperparameter tuning using GridSearchCV. The function returns the prediction error (mean squared error)
    and the best model trained on the current fold. 
    '''
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    skfold = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
    gridcv = GridSearchCV(pipeline, cv=skfold, n_jobs=1, param_grid=param_grid,
                          scoring='neg_mean_squared_error')
    
    if var_sel == 'No': 
        gridcv.fit(X_train_scaled, y_train)
    elif var_sel == 'ANOVA': 
        best_k = search_best_k(X_train_scaled, y_train)
        filtering = SelectKBest(f_regression, k=best_k)
        filtering.fit(X_train_scaled, y_train)
        X_train_sel = filtering.transform(X_train_scaled)
        X_test_sel = filtering.transform(X_test_scaled)
        gridcv.fit(X_train_sel, y_train)
    elif var_sel == 'RF':
        best_k = int(np.round(X.shape[1] * 0.2))
        filtering = SelectKBest(f_regression, k=best_k)
        filtering.fit(X_train_scaled, y_train)
        X_train_sel = filtering.transform(X_train_scaled)
        X_test_sel = filtering.transform(X_test_scaled)
        
        rf_selection = SelectFromModel(RandomForestRegressor(n_estimators=2000,
                                                             random_state=0),
                                       threshold=0.0)
        rf_selection.fit(X_train_sel, y_train)
        
        feature_importances = rf_selection.estimator_.feature_importances_
        if len(feature_importances) > 9:
            rf_selection.threshold = -1.0 * np.sort(-1.0 * feature_importances)[9]
        else:
            rf_selection.threshold = -1.0 * np.sort(-1.0 * feature_importances)[-1]
        
        X_train_sel = rf_selection.transform(X_train_sel)
        X_test_sel = rf_selection.transform(X_test_sel)
        gridcv.fit(X_train_sel, y_train)
    
    best_model = gridcv.best_estimator_
    if var_sel == 'No': 
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)
    else: 
        best_model.fit(X_train_sel, y_train)
        y_pred = best_model.predict(X_test_sel) 
       
    pred_error = mean_squared_error(y_test, y_pred)
    return pred_error, best_model

def search_best_k(X_train_scaled, y_train):
    '''This function searches for the best number of features (k) for feature selection using ANOVA F-test. 
    It evaluates different values of k (5, 10, 15, 20, 25, 30, 35) by applying K-Nearest Neighbors and cross-validation,
    and returns the value of k that minimizes the mean squared error.
    '''
    k_values = [5, 10, 15, 20, 25, 30, 35]
    best_k_score = float('inf')
    best_k_value = k_values[0]
    
    for k in k_values:
        filtering = SelectKBest(f_regression, k=k)
        X_k_selected = filtering.fit_transform(X_train_scaled, y_train)
        
        knn_model = KNeighborsRegressor()
        scores = cross_val_score(knn_model, X_k_selected, y_train,
                                 cv=RepeatedKFold(n_splits=5,
                                                  n_repeats=1,
                                                  random_state=0),
                                 scoring='neg_mean_squared_error')
        
        mean_score = np.mean(scores)
        
        if mean_score < best_k_score:
            best_k_score = mean_score
            best_k_value = k
    
    return best_k_value
    
def evaluate_dummy_regressor(X, y, n_repeats=10, strat='mean'):
    """
    Evaluate the performance of a Dummy Regressor using repeated k-fold cross-validation.

    Parameters:
    - X: Input features.
    - y: Target variable.
    - n_repeats: Number of times cross-validation is repeated.
    - strategy: Strategy to use for the dummy regressor. Options are 'mean', 'median', 'quantile', 'constant'.

    Returns:
    - average_pred_error: Average prediction error (mean squared error) across all outer folds and repeats.
    """
    outer_cv = RepeatedKFold(n_splits=10, n_repeats=n_repeats, random_state=0)
    pred_error_list = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        dummy_regressor = DummyRegressor(strategy=strat)
        dummy_regressor.fit(X_train, y_train)
        y_pred = dummy_regressor.predict(X_test)
        
        pred_error = mean_squared_error(y_test, y_pred)
        pred_error_list.append(pred_error)

    average_pred_error = np.mean(pred_error_list)
    sd_pred_error = np.std(pred_error_list)
    return average_pred_error,sd_pred_error,dummy_regressor

def x_y_extraction_models(df_X,df_Y,water=False):
    '''
    This function extracts features from two dataframes (df_X and df_Y), merges them based on the 'id' column, 
    and prepares the data for machine learning models. It handles feature selection and extraction using the MinimalFCParameters 
    from the tsfresh package, with optional inclusion of water-related features. It returns the extracted features (X) 
    and the corresponding Gibbs free energy (dG) values (Y), along with additional information about residue names and ids.
    '''
    merged_df = pd.merge(df_X,df_Y, on='id', suffixes=('', '_drop'))
    merged_df.drop([col for col in merged_df.columns if 'drop' in col], axis=1, inplace=True)
    minimal_params = MinimalFCParameters()
    del minimal_params['length']
    if water==False:
        features_final= merged_df.iloc[:,[0,5,6,7,9,10,11,12]]
    else:
        features_final= merged_df.iloc[:,[0,5,6,7,8,9,10,11,12]]

    X = extract_features(features_final,column_id="id", column_sort="frame", column_kind=None, column_value=None,default_fc_parameters=minimal_params)
    Y_values=merged_df.iloc[:,[0,13]].drop_duplicates()
    Y_values = Y_values.set_index('id')
    ordered_data = Y_values.loc[X.index]
    ordered_data['id']=ordered_data.index
    Y_id=ordered_data
    Y_id['res_name'] = Y_id['id'].str.extract(r'_([A-Z]{3})(?=\d+)')
    cols=['id','res_name','dG']
    Y_id=Y_id[cols]
    return X,Y_id['dG'],Y_id



def models_null(df_Y):
    '''
    This function evaluates simple machine learning models (using K-Nearest Neighbors) on a dataset of protein residues, 
    based on various properties such as weight, hydrophobicity, acceptor, and donor characteristics. The function maps these 
    properties to the residue names in the dataset and evaluates the prediction error for each model. It returns a list 
    of evaluation results including the model name, feature set, and the corresponding prediction errors.

    The following properties are considered for each residue:
     - Weight: Molecular weight of the amino acid.
     - Hydrophobicity: The hydrophobicity of each amino acid.
    '''
    Y_final_res=df_Y#.iloc[:,[0,3,5]] #takes also res column
    weight_dict = {
        'ALA': 89.0, 'ARG': 174.0, 'ASN': 132.0, 'ASP': 133.0, 'CYS': 121.0, 'GLN': 146.0, 'GLU': 147.0, 'GLY': 75.0,
        'HIS': 155.0, 'ILE': 131.0, 'LEU': 131.0, 'LYS': 146.0, 'MET': 149.0, 'PHE': 165.0, 'PRO': 115.0, 'SER': 105.0,
        'THR': 119.0, 'TRP': 204.0, 'TYR': 181.0, 'VAL': 117.0
    }

    
    hydrophobicity_dict = {
        'ALA': 51.0, 'ARG': -144.0, 'ASN': -84.0, 'ASP': -78.0, 'CYS': 137.0, 'GLN': -128.0, 'GLU': -115.0,
        'GLY': -13.0, 'HIS': -55.0, 'ILE': 106.0, 'LEU': 103.0, 'LYS': -205.0, 'MET': 73.0, 'PHE': 108.0,
        'PRO': -79.0, 'SER': -26.0, 'THR': -3.0, 'TRP': 69.0, 'TYR': 11.0, 'VAL': 108.0
    }
    # Addition of this properties for each aminoacid of the Y values
    Y_final_res.loc[:,'weight'] = Y_final_res['res_name'].map(weight_dict)
    Y_final_res.loc[:,'hydrophobicity'] = Y_final_res['res_name'].map(hydrophobicity_dict)
    data_models=[]
    Y = Y_final_res['dG']
    
    features_simple =pd.DataFrame(Y_final_res['weight']) 
    average_error,sd_error,model=evaluate_model(features_simple,Y, model_name='knn', n_repeats=10, var_sel='No')
    data_models.append(['null_weight',features_simple,Y,average_error,sd_error,model])
    
    features_simple =pd.DataFrame(Y_final_res['hydrophobicity']) 
    average_error,sd_error,model=evaluate_model(features_simple,Y,model_name='knn', n_repeats=10, var_sel='No')
    data_models.append(['null_hydrophobic',features_simple,Y,average_error,sd_error,model])
    

    return data_models


def models(X,Y,models_list,prot):
    '''
    This function evaluates multiple machine learning models on a given dataset (X and Y) using the evaluate_model function. 
    It iterates over a list of models (models_list), each of which is defined by a tuple containing the model name, 
    variable selection method, and a descriptive name. For each model, it computes the average and standard deviation 
    of the prediction error over multiple repetitions and returns the results in a list. The results include the model name, 
    features used, and the corresponding evaluation metrics.

    Arguments:
    - X: The feature matrix.
    - Y: The target variable.
    - models_list: A list of tuples containing model name, variable selection method, and a descriptive name.
    - prot: A string representing the protein or sample name for reference.

    Returns:
    - data_models: A list of evaluation results for each model.
    '''

    data_models=[]
    for model,varsel,name in models_list:
        avg_error,sd_error,model=evaluate_model(X,Y,model_name=model, n_repeats=10, var_sel=varsel)
        data_models.append([prot,name,X,Y,avg_error,sd_error,model])
    return data_models
