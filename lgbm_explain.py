# +
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

#sys.path.append( '/Users/meiyinlo/Documents/coding/ULINK/ai/common/ml_models/' )
#import lgbm_explain
# -

####lgb.create_tree_digraph(model, 0)
####model_trees.loc[model_trees.tree_index == i]

def load_model(file_name):
 # load
 with open(file_name, 'rb') as f:
    return pickle.load(f)
    
def get_model_and_trees(model_file_name):
    model = load_model(model_file_name)
    model_trees = model.booster_.trees_to_dataframe()
    return model, model_trees
    
def get_negative_decision_type(decision_type):
  assert decision_type in ['<=','>=']
  if decision_type == '<=':
    return '>'
  if decision_type == '>=':
    return '<'

def get_criteria_output_string(split_feature,decision_type, threshold, value ):
  return f"( {split_feature} {decision_type} {threshold} ) --> {value}"
def get_criteria_output_collection(split_feature,decision_type, threshold, value ):
  return {"split_feature": split_feature,
          "decision_type" :decision_type,
          "threshold": threshold ,
          "value": value }

def get_node_index( tree_index, leaf_index):
  return f"{tree_index}-L{leaf_index}"
def get_leaf_index(tree_index , node_index):
  return node_index.replace(f'{tree_index}-L','')
def get_node_row(model_trees, tree_index, node_index):
  return model_trees.loc[(model_trees.tree_index == tree_index) & (model_trees.node_index == node_index) ]

def get_criteria(model_trees, tree_index, node_index, output_function ):

  node_row = get_node_row(model_trees, tree_index, node_index)
  node_value = node_row['value'].values[0]
  parent_index = node_row['parent_index'].values[0]
  assert not( parent_index is None )
  parent_node_row = get_node_row(model_trees, tree_index, parent_index)
  split_feature = parent_node_row['split_feature'].values[0]
  decision_type = parent_node_row['decision_type'].values[0]

  threshold = parent_node_row['threshold'].values[0]
  left_child = parent_node_row['left_child'].values[0]
  right_child = parent_node_row['right_child'].values[0]
  if left_child == node_index :
    criteria = output_function(split_feature,decision_type, threshold, node_value) #f"( {split_feature} {decision_type} {threshold} )"
  if right_child == node_index :
    criteria = output_function(split_feature,get_negative_decision_type(decision_type), threshold, node_value) #f"( {split_feature} {get_negative_decision_type(decision_type)} {threshold} )"
  return  criteria

def get_nested_criteria(model_trees, tree_index, node_index, output_function = get_criteria_output_string, criteria_list=[]):
  node_row = get_node_row(model_trees, tree_index, node_index)
  parent_index = node_row['parent_index'].values[0]
  if parent_index is None:
    return criteria_list
  criteria = get_criteria(model_trees, tree_index, node_index,output_function)
  criteria_list = criteria_list + [criteria]
  return get_nested_criteria(model_trees, tree_index, parent_index, criteria_list=criteria_list,output_function=output_function)

def aggregate_criteria(criteria_df):
  criteria_df['max_threshold'] = criteria_df.groupby(['split_feature','decision_type'])['threshold'].transform('max')
  criteria_df['min_threshold'] = criteria_df.groupby(['split_feature','decision_type'])['threshold'].transform('min')
  criteria_df = criteria_df[['split_feature','decision_type','min_threshold','max_threshold']].drop_duplicates().reset_index(drop=True)
  assert len(  set(criteria_df.decision_type.unique().tolist() ) - set(['>','<=','<','>='])  ) == 0
  criteria_df['boundary_value'] = np.nan
  criteria_df.loc[criteria_df['decision_type'] == '>' ,'boundary_value'] = criteria_df['max_threshold']
  criteria_df.loc[criteria_df['decision_type'] == '<' ,'boundary_value'] = criteria_df['min_threshold']
  criteria_df.loc[criteria_df['decision_type'] == '>=' ,'boundary_value'] = criteria_df['max_threshold']
  criteria_df.loc[criteria_df['decision_type'] == '<=' ,'boundary_value'] = criteria_df['min_threshold']
  return criteria_df[['split_feature','decision_type','boundary_value']]
  
def get_tree_leaf_score(model_trees, tree_index,leaf_index):
  node_index = get_node_index( tree_index, leaf_index)
  node_row = get_node_row(model_trees, tree_index, node_index)
  node_value = node_row['value'].values[0]
  return node_value
  
def get_tree_leaf_score_flatten_criteria_df(model_trees):
  tree_index_list = model_trees.tree_index.unique().tolist()
  result = pd.DataFrame()
  for tree_index in tree_index_list :
    node_index_list = model_trees.loc[model_trees.tree_index == tree_index , 'node_index'].unique().tolist()
    leaf_node_index_list = [i for i in node_index_list if f"{tree_index}-L" in i]
    for leaf_node_index in leaf_node_index_list:
      leaf_index = get_leaf_index(tree_index , leaf_node_index)
      leaf_score = get_tree_leaf_score(model_trees, tree_index,leaf_index)
      criteria_df = aggregate_criteria( pd.DataFrame( get_nested_criteria(model_trees, tree_index, leaf_node_index, output_function = get_criteria_output_collection, criteria_list=[]) ) )
      criteria_df['tree_index'] = tree_index
      criteria_df['leaf_index'] = leaf_index
      criteria_df['leaf_score'] = leaf_score
      result = pd.concat([result, criteria_df], ignore_index=True)
  return result
    #get_leaf_index(tree_index , node_index):
    #print(tree_index, len(node_index_list),len(leaf_node_index_list))
    
def get_filtered_tree_leaf_score_flatten_criteria_df(tree_leaf_score_flatten_criteria_df, feature_list ):
  tree_leaf_score_flatten_criteria_df['has_feature'] = 0
  tree_leaf_score_flatten_criteria_df.loc[tree_leaf_score_flatten_criteria_df.split_feature.isin(feature_list) , 'has_feature'] = 1
  tree_leaf_score_flatten_criteria_df['has_feature_max'] = tree_leaf_score_flatten_criteria_df.groupby(['tree_index','leaf_index'])['has_feature'].transform('max')
  tree_leaf_score_flatten_criteria_df = tree_leaf_score_flatten_criteria_df.loc[tree_leaf_score_flatten_criteria_df.has_feature_max == 1].reset_index(drop=True)
  return tree_leaf_score_flatten_criteria_df
  
def get_merged_column_data_criteria_df(model_trees, tree_index, leaf_index,column_data ):
  node_index = get_node_index( tree_index, leaf_index)
  criteria_df = pd.DataFrame(get_nested_criteria(model_trees, tree_index, node_index, criteria_list=[], output_function=get_criteria_output_collection) )
  criteria_df = aggregate_criteria(criteria_df)
  criteria_df = criteria_df.merge(column_data, how='left', left_on = ['split_feature'] , right_on = 'column_name')
  return criteria_df

def get_feature_name(model):
    if isinstance( model , lgb.Booster):
      return model.feature_name()
    else :
        return model.feature_name_

def get_booster(model):
    if isinstance( model , lgb.Booster):
      return model
    else :
        return model.booster_
    
def get_row_data_pred_history(model, row_data):
  assert len(row_data) == 1, len(row_data)
  model_trees = get_booster(model).trees_to_dataframe()
  column_data = row_data.reset_index(drop=True).iloc[0].T.reset_index().set_axis(['column_name', 'column_value'], axis=1)
  final_score = model.predict(row_data.loc[: , get_feature_name(model)].to_numpy() ,  pred_leaf = False , raw_score = True )[0]
  leaf_index_list = model.predict(row_data.loc[: , get_feature_name(model)].to_numpy() ,  pred_leaf = True , raw_score = True )[0]
  total_score = 0
  pred_history = pd.DataFrame()
  criteria_df = pd.DataFrame()
  for i in range(len(leaf_index_list)) :

   leaf_index = leaf_index_list[i]
   tree_score = get_tree_leaf_score(model_trees, i,leaf_index)
   total_score += tree_score
   pred_history = pd.concat([pred_history , pd.DataFrame(data={'tree_index' : [i] , 'leaf_index' : [leaf_index]
                                                               , 'tree_score' : [tree_score] , 'accumulated_score' : [total_score] })] , ignore_index=True)
   sub_criteria_df = get_merged_column_data_criteria_df(model_trees, i, leaf_index , column_data)
   sub_criteria_df['tree_index'] = i
   sub_criteria_df['tree_score'] = tree_score
   criteria_df = pd.concat([criteria_df , sub_criteria_df] , ignore_index=True)

  assert total_score == final_score
  pred_history['final_score'] = final_score

  return pred_history, criteria_df
  
def get_pred_history_df(df, model , df_filter_criteria):
  assert list(range(len(df))) == df.index.values.tolist()
  pred_history = pd.DataFrame()
  criteria_df = pd.DataFrame()
  record_df = pd.DataFrame()
  for i, row in df.loc[df_filter_criteria].iterrows():
   row['record_index'] = i
   record_df = pd.concat([record_df ,pd.DataFrame([row]) ], ignore_index = True)
   sub_pred_history, sub_criteria_df = get_row_data_pred_history(model, df.iloc[i:(i+1)])
   sub_pred_history['record_index'] = i
   sub_criteria_df['record_index'] = i
   pred_history = pd.concat([pred_history , sub_pred_history] , ignore_index=True)
   criteria_df = pd.concat([criteria_df , sub_criteria_df] , ignore_index=True)
   print(i)
  
  return  pred_history, criteria_df, record_df
  
def plot_pred_history_df(df, model , df_filter_criteria):
    pred_history, criteria_df, record_df = get_pred_history_df(df, model , df_filter_criteria)
    for record_index in pred_history.record_index.unique().tolist():
     sub_df = pred_history.loc[pred_history.record_index == record_index]
     plt.plot(sub_df['tree_index'] , sub_df['accumulated_score'] , label = f'{record_index}')
    plt.legend()
    
def plot_box_tree_leaf_split_feature_count(tree_leaf_score_flatten_criteria_df , ax=None):
    df = tree_leaf_score_flatten_criteria_df.groupby(['tree_index' , 'leaf_index'])['split_feature'].aggregate('nunique').reset_index()
    if ax is None :
      fig, ax = plt.subplots(figsize=(20,8))
    df.boxplot(column=['split_feature'], by='tree_index' , ax = ax )
    
def plot_hist_tree_split_feature_count(model_trees_to_dataframe , ax=None):
    df = model_trees_to_dataframe.groupby(['tree_index' ])['split_feature'].aggregate('nunique').reset_index()
    if ax is None :
      fig, ax = plt.subplots(figsize=(20,8))
    df['split_feature'].hist( ax = ax )
    
def get_tree_split_feature_avg(model_trees_to_dataframe , ax=None):
    df = model_trees_to_dataframe.groupby(['tree_index' ])['split_feature'].aggregate('nunique').reset_index()
    return np.mean(df['split_feature'])    
    
def get_used_features(model):
  #pd.set_option('display.max_rows', None)
  feature_importances = pd.DataFrame(data={'importance': model.feature_importances_ ,
                  'column_name' :get_feature_name(model)}).sort_values(by=['importance'], ascending=False)
  feature_importances['original_feature'] = feature_importances['column_name'].replace(regex=[r'_sum$',r'_diff$',r'_max$',r'_countd$',r'_min$',r'_positive_lag$',r'_negative_lag$'], value='' )
  feature_importances['original_feature_sum'] = feature_importances.groupby('original_feature')['importance'].transform('sum')          
  total_split_feature_count = len(feature_importances[feature_importances.importance	> 0 ])
  total_unused_original_features = len(feature_importances[feature_importances.original_feature_sum	== 0 ])                   
  unused_original_features = feature_importances[feature_importances.original_feature_sum	== 0 ]['original_feature'].unique().tolist()
  return total_split_feature_count, total_unused_original_features, unused_original_features
