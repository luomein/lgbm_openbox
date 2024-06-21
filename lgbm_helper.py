#import streamlit as st
import lightgbm as lgb
import io
import pandas as pd
import tempfile
import os

def get_model_write_to_temp_file(uploaded_model_file):
 if uploaded_model_file is not None:

    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the path to save the uploaded file
        temp_file_path = os.path.join(tmpdir, uploaded_model_file.name)
        
        # Save the file to the temporary directory
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_model_file.read())
        
        # Display the path of the temporary file
        #st.write(f"File saved to temporary path: {temp_file_path}")
        model = lgb.Booster(model_file=temp_file_path)
        #st.dataframe(data=model.trees_to_dataframe())
        return model


def get_model(uploaded_model_file):
 
 if uploaded_model_file is not None:
    model_bytes = io.BytesIO( uploaded_model_file.read() )
    #model = pickle.load(io.BytesIO(model_bytes))
    stringio = model_bytes.getvalue()
    stringio = stringio.decode("utf-8")
    #stringio = StringIO(uploaded_model_file.getvalue().decode("utf-8"))
    #stringio = StringIO(uploaded_model_file.getvalue())
    model = lgb.Booster(model_str=stringio)
    #st.dataframe(data=model.trees_to_dataframe())
    return model

 else:
     return None

def get_feature_name(model):
    bst =  get_booster(model)
    return bst.feature_name()
    #if isinstance( model , lgb.Booster):
    #  return model.feature_name()
    #else :
    #    return model.feature_name_

def get_booster(model):
    if isinstance( model , lgb.Booster):
      return model
    else :
        return model.booster_

def get_feature_infos(model):
    bst =  get_booster(model)
    model_details = bst.dump_model()
    feature_infos = model_details['feature_infos']
    return feature_infos

def get_feature_importance(model):
    bst =  get_booster(model)
    return bst.feature_importance()

def get_feature_summary_df(model):
    feature_importances = get_feature_importance(model)
    feature_names = get_feature_name(model)
    feature_infos = get_feature_infos(model)

    min_vals = []
    max_vals = []

    for feature in feature_names:
      min_val = feature_infos[feature]['min_value']
      max_val = feature_infos[feature]['max_value']

      min_vals.append(min_val)
      max_vals.append(max_val)

    # Create a DataFrame
    df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances,
    'Min Value': min_vals,
    'Max Value': max_vals
    })
    return df
def get_tree_summary(model):
    bst =  get_booster(model)
    model_details = bst.dump_model()
    selected_tree_columns = ['tree_index', 'num_leaves', 'num_cat', 'shrinkage' ]
    tree_info = [ {k : tree[k] for k in selected_tree_columns } for tree in model_details['tree_info'] ]
    tree_info = pd.DataFrame(tree_info)
    tree_detail = bst.trees_to_dataframe()
    tree_detail = tree_detail.groupby('tree_index').agg(max_node_depth = ('node_depth' , 'max')
                                                        , count_distinct_features = ('split_feature' , 'nunique')
                                                        , num_nodes=('node_index','nunique') 
                                                        , count_rows=('node_index','count')).reset_index(drop=False)
    assert len(tree_detail[tree_detail['num_nodes'] == tree_detail['count_rows']] ) == len(tree_detail)
    tree_detail = tree_detail.drop(columns = 'count_rows' )
    assert len(tree_detail) == len(tree_info)
    return tree_info.merge(tree_detail , on=['tree_index'])

def get_parameter_df(model):
    bst =  get_booster(model)
    model_details = bst.dump_model()
    model_configuration = dict()
    for k in bst.params.keys() :
       model_configuration[k] =  bst.params[k]
    for k in model_details.keys():
      if not k in ['tree_info' , 'feature_names' , 'feature_importances' , 'feature_infos'] :
      #print(k , model_details[k])
        model_configuration[k] = model_details[k]

    return pd.DataFrame([ model_configuration ]).T.reset_index(drop=False).set_axis(['key','value'], axis=1)

def get_node_row(model_trees, tree_index, node_index):
  return model_trees.loc[(model_trees.tree_index == tree_index) & (model_trees.node_index == node_index) ]

def get_criteria_output_string(split_feature,decision_type, threshold, value , node_index , include_na ):
  return f"( {node_index} {split_feature} {decision_type} {threshold} {'or na' if include_na else ''} ) --> {value}"
def get_criteria_output_collection(split_feature,decision_type, threshold, value , node_index , include_na ):
    return {"node_index" : node_index ,
          "split_feature": split_feature,
          "decision_type" :decision_type,
          "threshold": threshold ,
            "include_na" : include_na,
          "value": value }
def get_negative_decision_type(decision_type):
  assert decision_type in ['<=','>=']
  if decision_type == '<=':
    return '>'
  if decision_type == '>=':
    return '<'


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
  missing_type =  parent_node_row['missing_type'].values[0]
  missing_direction =  parent_node_row['missing_direction'].values[0]
  #assert ( missing_direction in ['left', 'right'] ) == ( missing_type == 'NaN' ) , f"{missing_direction} , {missing_type}"

  if left_child == node_index :
    include_na = (missing_direction == 'left')  
    criteria = output_function(split_feature,decision_type, threshold, node_value, node_index , include_na) #f"( {split_feature} {decision_type} {threshold} )"
  if right_child == node_index :
    include_na = (missing_direction == 'right')    
    criteria = output_function(split_feature,get_negative_decision_type(decision_type), threshold, node_value , node_index , include_na ) #f"( {split_feature} {get_negative_decision_type(decision_type)} {threshold} )"
  return  criteria

def get_nested_criteria(model_trees, tree_index, node_index, output_function = get_criteria_output_string, criteria_list=[]):
  node_row = get_node_row(model_trees, tree_index, node_index)
  parent_index = node_row['parent_index'].values[0]
  if parent_index is None:
    return criteria_list
  criteria = get_criteria(model_trees, tree_index, node_index,output_function)
  criteria_list = criteria_list + [criteria]
  return get_nested_criteria(model_trees, tree_index, parent_index, criteria_list=criteria_list,output_function=output_function)

def get_booster_nested_criteria(model_trees, tree_index_list , node_index_list , output_function=get_criteria_output_collection ):
    criteria_df = pd.DataFrame()
    assert len(tree_index_list)==len(node_index_list)
    for i in range(len(tree_index_list)):
      tree_index = tree_index_list[i]
      node_index = node_index_list[i]
      tree_criteria_df = pd.DataFrame( get_nested_criteria(model_trees, tree_index, node_index, output_function =output_function ) )
      tree_criteria_df['tree_index'] = tree_index
      #tree_criteria_df['node_index'] = node_index

      criteria_df = pd.concat([criteria_df ,tree_criteria_df ] , ignore_index=True )
    return criteria_df

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
      
