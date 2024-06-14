import streamlit as st
import lgbm_helper
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb


def model_txt_hint_expander():
  with st.expander("💡 How to export lgbm model to txt file"):
    st.write('lgbm model')
    code = '''
import lightgbm as lgbm


lgb_model = lgbm.LGBMRegressor()
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
lgb_model.booster_.save_model('model.txt')
'''
    st.code(code, language='python')

    st.write('lgbm model booster')
    code2 = '''
import lightgbm as lgbm

train_data = lgbm.Dataset(X_train, label=y_train)
test_data = lgbm.Dataset(X_test, label=y_test)

params = {'objective':'regression'}
lgb_model_booster = lgbm.train(params, train_set=train_data, valid_sets=[test_data])
lgb_model_booster.save_model('model_booster.txt')
'''
    st.code(code2, language='python')

def model_summary_features_df(model):
    if not model is None :
      st.dataframe(lgbm_helper.get_feature_summary_df(model), use_container_width=True)
    else:
        st.write('empty')

def model_summary_parameters_df(model):
    if not model is None :
      st.dataframe(lgbm_helper.get_parameter_df(model), use_container_width=True)
    else:
        st.write('empty')

def model_summary_tabs(model):
  st.write('Model Summary')  
  if model is None :
   model_parameters, model_features , model_trees = st.tabs(["Parameters" , "Features" , "Trees"])
   with model_parameters:
    st.write('empty')
   with  model_features :
    st.write('empty')
   with model_trees :
    st.write('empty')

  else:
      feature_summary_df = lgbm_helper.get_feature_summary_df(model)
      tree_summary_df = lgbm_helper.get_tree_summary(model)
      model_parameters, model_features , model_trees = st.tabs(["Parameters" , f"Features({len(feature_summary_df)})" , f"Trees({len(tree_summary_df)})"])

      with model_parameters:
        #st.write(dir(model_parameters))
        model_summary_parameters_df(model)
      with model_features:
        st.dataframe(feature_summary_df, use_container_width=True)
      with model_trees:
          st.dataframe(tree_summary_df, use_container_width=True)

def dataset_summary_tabs(df):
    st.write('Dataset Summary')
    if df is None :
        df_columns, df_records = st.tabs(["Columns" , "Records"])
        with df_columns:
            st.write('empty')
        with df_records :
            st.write('empty')
    else :
        df_columns, df_records = st.tabs([f"Columns({len(df.columns)})" , f"Records({len(df)})"])
        with df_columns:
            describe_df = pd.DataFrame(df.describe()).T.rename_axis('column').reset_index(drop=False)[['column','min','max']]
            df = pd.DataFrame(df.columns.tolist() , columns=['column']) 
            df = df.merge(describe_df , on='column', how='left')
            st.dataframe(df , use_container_width=True)
            #st.dataframe( pd.DataFrame(df.columns.tolist() , columns=['column']) , use_container_width=True)
        with df_records :
            st.dataframe( df , use_container_width=True)

def dataset_validation(df,model):
    if df is None or model is None :
        st.write('empty')
    else:
        if len(df) == 0 :
            st.write('⚠️  Dataset is empty')
            return False
        if set(lgbm_helper.get_feature_name(model)) <= set(df.columns.tolist()):
            st.write('✅ Dataset columns match model features.')
            return True
        else:    
            st.write("⚠️  Dataset columns do not match model features")
            missing_columns = list(set(lgbm_helper.get_feature_name(model)) - set(df.columns.tolist()))
            st.write(f"Missing columns: [{','.join(missing_columns)}]")
    return False

def show_prediction(df,model,valid):
    if df is None or model is None or not valid :
        st.write('empty')
        return False
    else:
        prediction = model.predict(df[lgbm_helper.get_feature_name(model)])
        #st.write(prediction)
        #st.dataframe(prediction)
        st.dataframe( pd.DataFrame(prediction , columns=['prediction']).rename_axis('record_index') ,  use_container_width=True)
        return True

def show_booster_detail(df,model,show_prediction):
    
    if df is None or model is None or not show_prediction :
        st.header('Booster Detail', anchor = 'booster') 
        st.write('empty')
        st.header('Individual Tree Detail', anchor = 'tree') 
        st.write('empty')

        return None , None
        #return False
    else:
       st.header('Booster Detail', anchor = 'booster') 
       bst =  lgbm_helper.get_booster(model)
       tree_detail = bst.trees_to_dataframe()

       tree_index = st.slider("tree_index", tree_detail.tree_index.min(), tree_detail.tree_index.max() )
       if len(df) > 1:
         target_index = st.slider("record_index" , 0 , len(df) - 1 )
       else:
         target_index = 0
         st.write(f"record_index: {target_index}")

       leaf_indices = bst.predict(df.iloc[target_index : (target_index + 1)][lgbm_helper.get_feature_name(model)],  pred_leaf=True)[0]
       record_df = df.iloc[target_index: (target_index + 1)][lgbm_helper.get_feature_name(model)]
       prediction = bst.predict( record_df ,  pred_leaf=False)[0]

       leaf_indices = pd.DataFrame(data={'tree_index' : list(range(len(leaf_indices))) , 
                                         'leaf_index' : leaf_indices 
                                         })
       leaf_indices['node_index'] = leaf_indices.apply(lambda row: f"{row['tree_index']}-L{row['leaf_index']}", axis=1)
       leaf_output = tree_detail.merge(leaf_indices , on = ['tree_index' , 'node_index'] , how = 'inner')
       leaf_output['prediction'] = leaf_output['value'].cumsum()
       criteria_df = lgbm_helper.get_booster_nested_criteria(tree_detail ,leaf_indices.tree_index.values.tolist() , leaf_indices.node_index.values.tolist() )

       booster_prediction , split_features = st.tabs(['Booster Prediction', 'Split Features'])
       with booster_prediction:
         plot_booster_prediction(leaf_output , tree_index)
       with split_features:
           #st.dataframe(criteria_df)
           show_split_features(df , target_index , criteria_df )
       st.header('Individual Tree Detail', anchor = 'tree') 
       tree_split_features ,tree_digraph = st.tabs([ 'Tree Split Features' , 'Tree Path'])
       with tree_split_features :
           #st.write('test')
           st.dataframe(criteria_df.loc[criteria_df.tree_index == tree_index , [ 'tree_index' ,'node_index' , 'split_feature','decision_type','threshold'] ].set_index('tree_index') )

       with tree_digraph:
         st.graphviz_chart( lgb.create_tree_digraph( model, tree_index , example_case = record_df ))


       return tree_index, target_index
   
def show_split_features(df , record_index , criteria_df ):
    for f in criteria_df.split_feature.unique().tolist():
        with st.expander(f"{f}: {df.iloc[record_index][f]}"):
          st.dataframe(criteria_df.loc[criteria_df.split_feature == f , [ 'tree_index' ,'node_index' , 'split_feature','decision_type','threshold'] ].set_index('tree_index') )

def plot_booster_prediction(leaf_output , tree_index):

       fig2 = go.Figure(go.Waterfall(
         name="booster prediction",
         orientation="v",
    measure= ['relative'] * len(leaf_output),
    x=leaf_output['tree_index'],
    #textposition="outside",
    #text=[str(val) for val in values],
    y=leaf_output['value'],
    connector={"line":{"color":"rgb(63, 63, 63)"}},
))
       fig2.add_vline(x=tree_index, opacity=0.2 )
       fig2.add_hline(y=leaf_output.loc[leaf_output.tree_index == tree_index , 'prediction' ].values[0] , opacity=0.2 ) 
       
       fig2.update_layout(
    #title="booster prediction",
    xaxis_title="tree_index",
    yaxis_title="prediction",
    showlegend=False
)

       st.plotly_chart(fig2, theme="streamlit", use_container_width=True)


