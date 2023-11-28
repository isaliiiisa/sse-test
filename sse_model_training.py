import pandas as pd
import numpy as np
import random

from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.cluster import KMeans

import pickle
import logging

import shap

outcome_metrics_list = ['speed', 'quality', 'efficiency', 'team_health']


def preliminary_stats_capabilities_df(df_cap):
    n_teams      = df_cap.team_id.nunique()
    n_L4_metrics = df_cap.l4.nunique()
    n_L3_metrics = df_cap.l3.nunique()
    
    missing_vals = df_cap.isna().sum()
    return "some df stats to print"



def preliminary_stats_outcomes_df(df_out):
    n_teams      = df_cap.team_id.nunique()
    ### Check for missing outcomes 
    missing_vals = df_cap.isna().sum()
    ### Message that the model would be trained only on existing values
    return "some df stats to print"



def read_capabilities_df(data):
    df_cap = pd.read_excel(data)
    df_cap.columns = [str(c).strip().lower().replace(" ", "_") for c in df_cap.columns]
    
    cap_metrics_list = ['team_id', 'l4', 'value']
    
    for cap_metric in cap_metrics_list:
        if cap_metric not in df_cap.columns:
            return f'{cap_metric} not in Capabilities dataframe!', ''
    return "All good!", df_cap


    
def read_outcomes_df(data, outcome_metrics_list=outcome_metrics_list):
    df_out = pd.read_excel(data)
    df_out.columns = [str(c).strip().lower().replace(" ", "_") for c in df_out.columns]
    
    for outcome_metric in outcome_metrics_list + ['team_id']:
        if outcome_metric not in df_out.columns:
            return f'{outcome_metric} not in Outcomes dataframe!', ''
    return "All good!", df_out


    
def outcomes_capabilities_match(df_cap, df_out):
    ### Check if capabilities data matches teams data
    cap_teams = set(df_cap.team_id.unique())
    out_teams = set(df_out.team_id.unique())
    
    if len(cap_teams - out_teams)>0:
        return f"Teams {', '.join(cap_teams - out_teams)} are missing in Capabilities dataframe, but present in Outcomes data", False
        
    if len(out_teams - cap_teams)>0:
        return f"Teams {', '.join(out_teams - cap_teams)} are missing in Outcomes dataframe, but present in Capabilities data", False
    
    if len(out_teams - cap_teams)==0:
        return "Outcome teams match capabilities teams", True

        
def template_capabilities_match(df_cap, levels_mapping_d):
    capabilities_l4 = set(df_cap.l4.unique())

    not_covered_m = capabilities_l4 - set(levels_mapping_d['l4_l3_mapping'].keys())
    if len(not_covered_m)>0:
        return f"{','.join(not_covered_m)} metrics are not covered by template", False
    else:
        return "All metrics are covered by template", True
    
    
    
def prepare_data_for_modelling(df_pivot, df_outcome):
    if df_outcome.shape[1] == 2:
        outcome_metric = df_outcome.drop('team_id', axis=1).columns[0]
    
        df_outcome = df_outcome.dropna()
        df_full = df_pivot.merge(df_outcome, how='inner', on='team_id').sort_values(by='team_id')
        n_teams = df_full.team_id.nunique()
        
        X = df_full .drop(['team_id', outcome_metric], axis=1)
        y = df_full[[outcome_metric]]
        
        logging.info(f'Prepared data for {outcome_metric} modelling')
        return X, y, n_teams
    

    
def train_model(X, y):
    # Split data into training and validation sets
    X_train, X_validation, y_train,  y_validation = train_test_split(X, y, test_size=0.1, random_state=42)
    param_grid = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    }
    
    model = CatBoostRegressor()
    logging.info('Start model training...')
    grid_search_result = model.grid_search(param_grid, cv=5, X=X_train, y=y_train)
    logging.info('Model trained successfully!')
    
    ### validation results 
    rmse_val = np.sqrt(mean_squared_error(y_validation, model.predict(X_validation)))
    r2_val = r2_score(y_validation, model.predict(X_validation))
    print(f"Validation accuracy: R^2 {r2_val}, RMSE {rmse_val}")
    
    return model, rmse_val, r2_val



def save_trained_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    
    
    
def assign_grades(numbers, n_grades):
    # Converting the list of numbers to a 2D array
    data = np.array(numbers).reshape(-1, 1)
    
    # Create a KMeans instance with n clusters
    kmeans = KMeans(n_clusters=n_grades)
    
    # Fit the model and predict the cluster labels
    kmeans.fit(data)
    labels = kmeans.predict(data)
    cluster_means = [data[labels == i].mean() for i in range(n_grades)]

    # Rank the clusters and create a dictionary for label assignment
    ranked_clusters = np.argsort(cluster_means)[::-1]
    label_dict = {ranked_clusters[i]: chr(65 + i) for i in range(n_grades)}

    # Assign labels to each number
    labeled_numbers = [(num, label_dict[label]) for num, label in zip(numbers, labels)]
    return [l for number, l in labeled_numbers]



def save_feature_importance(model, filename, n_grades):
    ### TO DO: rewrite with SHAP values
    feature_importances = model.get_feature_importance(prettified=True)
    feature_importances['grade'] = assign_grades(feature_importances.Importances, n_grades)
    feature_importances.to_excel(filename, index=False)

    
def save_model(model, outcome_metric, core):
    if core:
        model_filename =  f'core_sse_{outcome_metric.replace(" ", "_")}_model_trained.sav'
    else:
        model_filename =  f'temp_sse_{outcome_metric.replace(" ", "_")}_model_trained.sav'
        
    save_trained_model(model, model_filename) 
        
    
    
def model_training_workflow(df_pivot, df_out, outcome_metric, core):
    X, y, n_teams = prepare_data_for_modelling(df_pivot, df_out)

    model, rmse_val, r2_val  = train_model(X, y)

    save_model(model, outcome_metric, core)
    
    f_importances_filename =  f'core_sse_{outcome_metric.replace(" ", "_")}_feature_importances.xlsx'
    save_feature_importance(model, f_importances_filename, n_grades=4)

    
    model_stats = {'training_sample_size':n_teams,
                   'RMSE_validation':rmse_val,
                   'R2_validation':r2_val,
                   }   
    return model_stats  