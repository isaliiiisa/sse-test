import pandas as pd
import numpy as np


def read_data_template(data):
    template_df = pd.read_excel(data)
    template_df.columns = [str(c).strip().lower().replace(" ", "_") for c in template_df.columns]
    
    levels_mapping_d = {}
    for i in range(1,4):
        levels_mapping_d[f'l{i+1}_l{i}_mapping'] = dict(zip(template_df[f'l{i+1}'], template_df[f'l{i}']))
    levels_mapping_d['l1_l0_mapping'] = {'Practices':'', 'Tools':'', 'Talent':''}
    
    weights_mapping_d = {}
    for i in range(2,5):
        weights_mapping_d[f'l{i}_weight'] = dict(zip(template_df[f'l{i}'], template_df[f'l{i}_weight']))
        
    weights_mapping_d['l1_weight'] = {'Practices':1, 'Tools':1, 'Talent':1}

    value_range_d = dict(zip(template_df.l4, template_df.value_range))
    direction_d = dict(zip(template_df.l4, template_df.direction)) 
    return levels_mapping_d, weights_mapping_d, value_range_d, direction_d



def perc_to_score(perc, ascending=True):
    if ascending: 
        return float(perc)/ 100 
    else:
        return (100-float(perc))/100

    
    
def rank_to_score(rank, ascending=True):
    if ascending:
        predefined_ranks = ['A', 'B', 'C', 'D', 'E'][::-1]
    else:
        predefined_ranks = ['A', 'B', 'C', 'D', 'E']        
    return (predefined_ranks.index(rank) + 0.5) / len(predefined_ranks)   



def int_to_score(num, ascending=True):
    if ascending: 
        return min(max((float(num) - 0) / (200 - 0), 0), 1)
    else:
        return min(max((float(num) - 200) / (0 - 200), 0), 1)
    
    
   
    
def calculate_L4_scores(df, value_col, range_col='value_range', direction='direction'):
    df['l4_score'] = None

    df.loc[df[range_col]=='A-E', 'l4_score'] = df.loc[df[range_col]=='A-E'].apply(lambda x: rank_to_score(x[value_col], x[direction]), axis=1)

    df.loc[df[range_col]=='0-100', 'l4_score'] = df.loc[df[range_col]=='0-100'].apply(lambda x: perc_to_score(x[value_col], x[direction]), axis=1)

    df.loc[~df[range_col].isin(['0-100', 'A-E']), 'l4_score'] = df.loc[~df[range_col].isin(['0-100', 'A-E'])].apply(lambda x: int_to_score(x[value_col], x[direction]), axis=1)
    return df



def aggregate_L3_scores(df, team_col=None):
    ### Calculate L3 scores based on L4 scores and L4 weights 
    df['l4_weight'] = df.groupby(['l3'])['l4_weight'].transform(lambda x: x/x.sum())
    
    ### Weight score by L4 weights 
    df['l4_weighted_score'] = df['l4_weight'] * df['l4_score']
        
    if team_col:
        df_agg = df.groupby(['team_id','l3']).agg({'l4_weighted_score':'sum'}).\
    reset_index().rename(columns={'l4_weighted_score':'l3_score'})
        
        df_pivot = df_agg.pivot_table(columns='l3', index=team_col, values='l3_score')
    else:
        df_agg = df.groupby(['l3']).agg({'l4_weighted_score':'sum'}).\
    reset_index().rename(columns={'l4_weighted_score':'l3_score'})
        
        df_pivot = df_agg.pivot_table(columns='l3', values='l3_score')
    return df_pivot



def assign_weights_levels(df, levels_mapping_d, weights_mapping_d, value_range_d, direction_d):
    for i in range(3,0,-1):
        df[f'l{i}'] = df[f'l{i+1}'].map(levels_mapping_d[f'l{i+1}_l{i}_mapping'])
        
    
    for i in range(1, 5):
        df[f'l{i}_weight'] = df[f'l{i}'].map(weights_mapping_d[f'l{i}_weight'])
        
    df['direction']   = df.l4.map(direction_d)
    df['value_range'] = df.l4.map(value_range_d)
    return df



def select_values_subset(sample_df, value_col, team_col=None):
    level_columns = ['l1', 'l2', 'l3', 'l4', 
                     'l1_weight', 'l2_weight', 'l3_weight', 'l4_weight', 
                     'direction', 'value_range']
    if team_col:
        level_columns.append(team_col)
    return sample_df[level_columns + [value_col]]



def preprocessing_pipeline(df, value_col, team_col=None):
    
    df_subset = select_values_subset(df, value_col,team_col)
    df_subset_scores = calculate_L4_scores(df_subset, value_col)
    df_agg = aggregate_L3_scores(df_subset_scores, team_col)

    return df_agg