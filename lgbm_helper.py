#import streamlit as st
import lightgbm as lgb
import io
import pandas as pd


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
    tree_detail = model.trees_to_dataframe()
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
    for k in model_details.keys():
      if not k in ['tree_info' , 'feature_names' , 'feature_importances' , 'feature_infos'] :
      #print(k , model_details[k])
        model_configuration[k] = model_details[k]

    return pd.DataFrame([ model_configuration ]).T.reset_index(drop=False).set_axis(['key','value'], axis=1)
